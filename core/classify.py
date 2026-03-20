
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import (
    AutoConfig,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
    ViTForImageClassification,
    ViTImageProcessor,
)

# Minimum crop dimension in pixels. Crops smaller than this are skipped so the
# classifier sees the full image instead of a tiny blurry patch.
_MIN_CROP_PX = 32

_DEFAULT_LABELS_MODEL = "nateraw/food"
_DEFAULT_CLIP_VISION_MODEL = "tanganke/clip-vit-large-patch14_food101"
_DEFAULT_CLIP_BASE_MODEL = "openai/clip-vit-large-patch14"
_DEFAULT_CLIP_PROMPT_TEMPLATE = "a photo of {}"

# 10-template prompt ensemble — averaged at init, zero inference-time cost.
# Research: Radford et al. 2021 shows ~5.6% zero-shot accuracy gain from ensembling.
# The CLIP paper's own food-domain template uses "a type of food" suffix (+~2% on Food-101).
# Expanded from 5 → 10 templates for broader semantic coverage.
_CLIP_PROMPT_TEMPLATES = [
    "a photo of {}, a type of food",        # CLIP paper's recommended food-domain template
    "a photo of {}",                         # generic baseline
    "a close-up photo of {}",               # matches zoomed bbox crops
    "a restaurant photo of {}",             # plated food context
    "a food photo of {}",                   # explicit food domain anchor
    "an overhead photo of {}",              # top-down plate photos
    "a photo of a dish called {}",          # names the dish explicitly
    "a photo of homemade {}",              # home cooking variation
    "a photo of {} on a plate",             # plating context
    "the food {}",                          # minimal context, competitive on short names
]

# Anchor prompts for the food/non-food binary gate (Change 6).
_FOOD_ANCHOR_PROMPTS = [
    "a photo of food",
    "a plate of food",
    "a close-up of a meal",
    "cooked food on a plate",
    "food ready to eat",
]
_NONFOOD_ANCHOR_PROMPTS = [
    "an empty plate",
    "a napkin",
    "a tablecloth",
    "cutlery",
    "an empty bowl",
    "a person's hand",
    "a cup",
]


class FoodClassifier:
    """
    Food-101 classifier.

    Default backend is CLIP zero-shot with fine-tuned Food-101 vision weights.
    Falls back to ViT classifier if CLIP setup fails.
    """

    def __init__(self, model_name: str = _DEFAULT_LABELS_MODEL):
        self.labels_model = model_name
        self.topk = max(3, int(os.getenv("CLASSIFIER_TOPK", "3")))
        self.backend = "none"

        self.device = self._choose_device(os.getenv("CLASSIFIER_DEVICE", "auto"))
        print(f"Loading classifier on device: {self.device}")

        classifier_backend = os.getenv("CLASSIFIER_BACKEND", "clip").strip().lower()
        if classifier_backend not in {"clip", "vit"}:
            classifier_backend = "clip"

        self.clip_processor: CLIPProcessor | None = None
        self.clip_model: CLIPModel | None = None
        self.text_features: torch.Tensor | None = None
        self.clip_labels: list[str] = []
        self.clip_prompt_template = os.getenv(
            "CLIP_PROMPT_TEMPLATE", _DEFAULT_CLIP_PROMPT_TEMPLATE
        )

        self.vit_processor: ViTImageProcessor | None = None
        self.vit_model: ViTForImageClassification | None = None

        # Test-Time Augmentation: average logits over N augmented crops.
        # Set CLASSIFIER_TTA=1 to enable (adds ~5× inference time per crop).
        self.use_tta: bool = os.getenv("CLASSIFIER_TTA", "0").strip() in {"1", "true", "yes"}

        if classifier_backend == "clip":
            try:
                self._init_clip()
            except Exception as exc:
                print(f"WARNING: CLIP classifier failed to initialize: {exc}")
                print("Falling back to ViT classifier.")
                self.clip_processor = None
                self.clip_model = None
                self.text_features = None
                self.clip_labels = []
                if self.device.type == "cuda":
                    if "out of memory" in str(exc).lower():
                        self.device = torch.device("cpu")
                        print("Retrying fallback backend on CPU after CUDA OOM.")
                    torch.cuda.empty_cache()
                self._init_vit(model_name)
        else:
            self._init_vit(model_name)

        print(f"Classifier backend ready: {self.backend}")

    @staticmethod
    def _choose_device(requested: str) -> torch.device:
        mode = requested.strip().lower()
        if mode == "cpu":
            return torch.device("cpu")
        if mode == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def normalize_label(label: str) -> str:
        """Normalize Food-101 labels for downstream matching."""
        return label.replace("_", " ").lower().strip()

    def _init_vit(self, model_name: str) -> None:
        print(f"Loading ViT classifier: {model_name}")
        self.vit_processor = ViTImageProcessor.from_pretrained(model_name)
        self.vit_model = ViTForImageClassification.from_pretrained(model_name).to(self.device).eval()
        self.backend = "vit"

    def _init_clip(self) -> None:
        clip_labels_model = os.getenv("CLIP_LABELS_MODEL", self.labels_model)
        clip_vision_model = os.getenv("CLIP_VISION_MODEL", _DEFAULT_CLIP_VISION_MODEL)
        clip_base_model = os.getenv("CLIP_BASE_MODEL", _DEFAULT_CLIP_BASE_MODEL)

        print("Loading CLIP classifier...")
        print(f"  labels model : {clip_labels_model}")
        print(f"  vision model : {clip_vision_model}")
        print(f"  base model   : {clip_base_model}")

        self.clip_labels = self._load_food101_labels(clip_labels_model)

        # Append extra labels from an optional file (regional / trending foods).
        extra_path = Path(os.getenv("CLIP_EXTRA_LABELS_FILE", "data/extra_clip_labels.txt"))
        if extra_path.exists():
            existing = set(self.clip_labels)
            with extra_path.open("r", encoding="utf-8") as fh:
                extra = [
                    self.normalize_label(ln)
                    for ln in fh
                    if ln.strip() and not ln.startswith("#")
                ]
            self.clip_labels += [lb for lb in extra if lb and lb not in existing]
            print(f"  extra labels loaded: {len(extra)} → {len(self.clip_labels)} total")

        prompts = [self.clip_prompt_template.format(label) for label in self.clip_labels]
        if not prompts:
            raise RuntimeError("No labels available for CLIP zero-shot classification.")

        self.clip_processor = CLIPProcessor.from_pretrained(clip_base_model, use_fast=False)
        self.clip_model = CLIPModel.from_pretrained(clip_base_model).to(self.device).eval()

        # Keep fine-tuned vision weights on CPU while loading to avoid doubling
        # GPU memory usage during startup.
        ft_vision = CLIPVisionModel.from_pretrained(clip_vision_model).eval()
        vision_state = ft_vision.state_dict()
        if vision_state and next(iter(vision_state.keys())).startswith("vision_model."):
            vision_state = {k.replace("vision_model.", "", 1): v for k, v in vision_state.items()}

        load_result = self.clip_model.vision_model.load_state_dict(vision_state, strict=False)
        critical_missing = [k for k in load_result.missing_keys if not k.endswith("position_ids")]
        critical_unexpected = [
            k for k in load_result.unexpected_keys if not k.endswith("position_ids")
        ]
        if critical_missing or critical_unexpected:
            raise RuntimeError(
                "Vision state mismatch while loading fine-tuned CLIP weights "
                f"(missing={len(critical_missing)}, unexpected={len(critical_unexpected)})."
            )

        # Precompute normalized text features once at startup — prompt ensembling.
        # Average over 5 templates: averaging unit vectors requires renorm after mean.
        use_ensemble = os.getenv("CLIP_PROMPT_ENSEMBLE", "1").strip() in {"1", "true", "yes"}
        templates = _CLIP_PROMPT_TEMPLATES if use_ensemble else [self.clip_prompt_template]
        all_feats: list[torch.Tensor] = []
        with torch.inference_mode():
            for tmpl in templates:
                t_prompts = [tmpl.format(lbl) for lbl in self.clip_labels]
                t_inp = self.clip_processor(text=t_prompts, padding=True, return_tensors="pt")
                t_inp = {k: v.to(self.device) for k, v in t_inp.items()}
                t_out = self.clip_model.text_model(**t_inp)
                t_feat = self.clip_model.text_projection(t_out.pooler_output)
                t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
                all_feats.append(t_feat)
            stacked = torch.stack(all_feats, dim=0).mean(dim=0)
            self.text_features = stacked / stacked.norm(dim=-1, keepdim=True)
        if use_ensemble:
            print(f"  Prompt ensemble: {len(templates)} templates × {len(self.clip_labels)} labels")

        # Precompute food/non-food binary gate anchor embeddings.
        with torch.inference_mode():
            def _enc_anchors(anchor_prompts: list[str]) -> torch.Tensor:
                inp = self.clip_processor(
                    text=anchor_prompts, padding=True, return_tensors="pt"
                )
                inp = {k: v.to(self.device) for k, v in inp.items()}
                out = self.clip_model.text_model(**inp)
                f = self.clip_model.text_projection(out.pooler_output)
                return f / f.norm(dim=-1, keepdim=True)
            self._food_anchor_feat    = _enc_anchors(_FOOD_ANCHOR_PROMPTS)    # (5, D)
            self._nonfood_anchor_feat = _enc_anchors(_NONFOOD_ANCHOR_PROMPTS) # (7, D)

        del ft_vision
        self.backend = "clip"

    def _load_food101_labels(self, labels_model: str) -> list[str]:
        try:
            cfg = AutoConfig.from_pretrained(labels_model)
            id2label = getattr(cfg, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                try:
                    ordered_items = sorted(id2label.items(), key=lambda kv: int(kv[0]))
                except Exception:
                    ordered_items = sorted(id2label.items(), key=lambda kv: str(kv[0]))
                labels = [self.normalize_label(str(v)) for _, v in ordered_items]
                if labels:
                    return labels
        except Exception:
            pass

        # Local fallback keeps the API usable without model hub access.
        db_path = Path("data/foods_database.json")
        if db_path.exists():
            with db_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            labels = sorted(str(k).strip().lower() for k in data.keys())
            if labels:
                return labels

        raise RuntimeError(
            "Could not load labels from CLIP_LABELS_MODEL or local data/foods_database.json."
        )

    def _safe_crop(self, image: Image.Image, bbox: Sequence[Any] | None) -> Image.Image:
        if not bbox or len(bbox) != 4:
            return image

        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            return image

        width, height = image.size
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        # PIL crop uses [left, upper, right, lower) so right/lower can equal
        # image width/height to include the last pixel row/column.
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return image

        if (x2 - x1) < _MIN_CROP_PX or (y2 - y1) < _MIN_CROP_PX:
            return image

        return image.crop((x1, y1, x2, y2))

    def _classify_vit(self, image: Image.Image) -> Dict:
        assert self.vit_processor is not None and self.vit_model is not None
        inputs = self.vit_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.vit_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            top_prob, top_idx = torch.topk(probs, min(self.topk, probs.shape[0]))

        topk = []
        for prob, idx in zip(top_prob, top_idx):
            raw_label = self.vit_model.config.id2label[idx.item()]
            label = self.normalize_label(raw_label)
            topk.append({"label": label, "confidence": float(prob.item())})

        return {"food_name": topk[0]["label"], "confidence": topk[0]["confidence"], "top3": topk}

    @staticmethod
    def _tta_variants(image: Image.Image) -> List[Image.Image]:
        """Return a list of augmented versions of image for TTA averaging."""
        w, h = image.size
        variants: List[Image.Image] = [image]  # original always first

        # Horizontal flip
        variants.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        # Brightness +10 %
        variants.append(ImageEnhance.Brightness(image).enhance(1.10))

        # Slightly darker (–10 %)
        variants.append(ImageEnhance.Brightness(image).enhance(0.90))

        # Centre crop 90 % then resize back
        crop_w, crop_h = int(w * 0.90), int(h * 0.90)
        left, top = (w - crop_w) // 2, (h - crop_h) // 2
        variants.append(
            image.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.BILINEAR)
        )

        # Slight contrast boost
        variants.append(ImageEnhance.Contrast(image).enhance(1.10))

        return variants

    def _image_to_features(self, image: Image.Image) -> torch.Tensor:
        """Encode one PIL image through the CLIP vision encoder; returns unit-normed features."""
        assert self.clip_processor is not None and self.clip_model is not None
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
            feat = self.clip_model.visual_projection(out.pooler_output)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat  # shape (1, D)

    def _classify_clip(self, image: Image.Image) -> Dict:
        assert (
            self.clip_processor is not None
            and self.clip_model is not None
            and self.text_features is not None
            and self.clip_labels
        )

        if self.use_tta:
            variants = self._tta_variants(image)
            all_probs = []
            with torch.inference_mode():
                scale = self.clip_model.logit_scale.exp()
                for aug in variants:
                    feat = self._image_to_features(aug)
                    logits = scale * (feat @ self.text_features.T)
                    all_probs.append(F.softmax(logits[0], dim=-1))
            probs = torch.stack(all_probs).mean(dim=0)
        else:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                image_outputs = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
                image_features = self.clip_model.visual_projection(image_outputs.pooler_output)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = self.clip_model.logit_scale.exp() * (image_features @ self.text_features.T)
                probs = torch.softmax(logits, dim=-1)[0]

        top_prob, top_idx = torch.topk(probs, min(self.topk, probs.shape[0]))
        topk = []
        for prob, idx in zip(top_prob, top_idx):
            label = self.clip_labels[int(idx.item())]
            topk.append({"label": label, "confidence": float(prob.item())})

        return {"food_name": topk[0]["label"], "confidence": topk[0]["confidence"], "top3": topk}

    def _switch_clip_to_cpu(self) -> None:
        if self.backend != "clip" or self.device.type == "cpu":
            return
        if self.clip_model is None or self.text_features is None:
            return
        self.device = torch.device("cpu")
        self.clip_model = self.clip_model.to(self.device)
        self.text_features = self.text_features.to(self.device)
        print("WARNING: CLIP classifier moved to CPU after CUDA OOM.")

    def classify_batch(
        self,
        images: list,           # List[Image.Image]
        bboxes: list = None,    # List[Sequence[Any] | None] — optional
    ) -> list:                  # List[Dict]
        """Run one CLIP forward pass for all crops at once.

        Falls back to per-image classify() calls when the ViT backend is active
        or when the image list is empty.
        """
        if not images:
            return []
        if bboxes is None:
            bboxes = [None] * len(images)
        if self.backend != "clip":
            return [self.classify(img, bbox) for img, bbox in zip(images, bboxes)]

        crops = [
            self._safe_crop(img.convert("RGB"), bbox)
            for img, bbox in zip(images, bboxes)
        ]
        assert (
            self.clip_processor is not None
            and self.clip_model is not None
            and self.text_features is not None
            and self.clip_labels
        )
        inputs = self.clip_processor(images=crops, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.inference_mode():
                img_out = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
                img_feat = self.clip_model.visual_projection(img_out.pooler_output)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = self.clip_model.logit_scale.exp() * (img_feat @ self.text_features.T)
                probs = torch.softmax(logits, dim=-1)
                top_prob, top_idx = torch.topk(probs, min(self.topk, probs.shape[-1]), dim=-1)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device.type == "cuda":
                torch.cuda.empty_cache()
                self._switch_clip_to_cpu()
                # Retry on CPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    img_out = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
                    img_feat = self.clip_model.visual_projection(img_out.pooler_output)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    logits = self.clip_model.logit_scale.exp() * (img_feat @ self.text_features.T)
                    probs = torch.softmax(logits, dim=-1)
                    top_prob, top_idx = torch.topk(probs, min(self.topk, probs.shape[-1]), dim=-1)
            else:
                raise

        results = []
        for i in range(len(crops)):
            topk = [
                {
                    "label": self.clip_labels[int(top_idx[i, j])],
                    "confidence": float(top_prob[i, j]),
                }
                for j in range(top_prob.shape[1])
            ]
            results.append({
                "food_name": topk[0]["label"],
                "confidence": topk[0]["confidence"],
                "top3": topk,
            })
        return results

    # ------------------------------------------------------------------
    # Hierarchical (coarse→fine) classification
    # ------------------------------------------------------------------

    # Groups of visually confusable foods.  When the top-1 label belongs to
    # one of these groups AND the confidence margin over top-2 is low, we run a
    # second CLIP pass restricted to just that group's labels — effectively
    # narrowing a 161-class problem to an 8-15-class one.
    _CONFUSABLE_GROUPS: Dict[str, List[str]] = {
        "burger": [
            "beef burger", "cheeseburger", "chicken burger", "veggie burger",
            "fish burger", "double cheeseburger", "hamburger",
        ],
        "pizza": [
            "pizza", "pepperoni pizza", "margherita pizza", "cheese pizza",
            "veggie pizza", "bbq chicken pizza",
        ],
        "soup": [
            "tomato soup", "chicken soup", "minestrone", "ramen", "pho",
            "miso soup", "clam chowder", "lentil soup", "french onion soup",
        ],
        "noodles": [
            "spaghetti", "spaghetti bolognese", "fettuccine alfredo", "pad thai",
            "ramen noodles", "udon", "soba", "lo mein", "chow mein",
        ],
        "curry": [
            "chicken curry", "beef curry", "lamb curry", "vegetable curry",
            "thai green curry", "tikka masala", "korma", "dal", "palak paneer",
        ],
        "salad": [
            "caesar salad", "greek salad", "garden salad", "pasta salad",
            "nicoise salad", "caprese salad", "coleslaw", "fruit salad",
        ],
        "rice": [
            "fried rice", "steamed rice", "biryani", "risotto", "pilaf",
            "congee", "onigiri",
        ],
        "sandwich": [
            "sandwich", "club sandwich", "blt", "grilled cheese", "panini",
            "sub sandwich", "wrap", "burrito", "taco",
        ],
        "cake": [
            "chocolate cake", "cheesecake", "carrot cake", "red velvet cake",
            "sponge cake", "lemon cake", "cupcake", "brownie",
        ],
        "egg": [
            "fried egg", "scrambled eggs", "boiled egg", "poached egg",
            "omelette", "eggs benedict",
        ],
        "chicken": [
            "fried chicken", "roast chicken", "chicken breast", "chicken wings",
            "chicken nuggets", "chicken tikka", "grilled chicken",
        ],
        "steak": [
            "steak", "beef steak", "ribeye steak", "sirloin steak",
            "pork chop", "lamb chop", "veal",
        ],
        "seafood": [
            "salmon", "tuna", "shrimp", "lobster", "crab", "fish and chips",
            "sushi", "sashimi", "grilled fish", "fish fillet",
        ],
        "bread": [
            "bread", "toast", "garlic bread", "croissant", "bagel",
            "naan", "pita", "flatbread", "baguette",
        ],
        "breakfast": [
            "pancakes", "waffles", "french toast", "granola",
            "overnight oats", "acai bowl", "smoothie bowl",
        ],
    }

    # Build reverse index: label → group key.
    _LABEL_TO_GROUP: Dict[str, str] = {}
    for _grp, _lbls in _CONFUSABLE_GROUPS.items():
        for _lbl in _lbls:
            _LABEL_TO_GROUP[_lbl] = _grp

    def _hierarchical_refine(self, image: Image.Image, initial: Dict) -> Dict:
        """Re-classify using a narrowed label set when top-1 is in a confusable group.

        Triggered only when confidence margin (top1 - top2) is below 0.25 AND
        the CLIP backend is active (ViT already has its own fine-grained head).
        """
        if self.backend != "clip" or self.text_features is None:
            return initial

        top3 = initial.get("top3", [])
        if len(top3) < 2:
            return initial

        top1_label = top3[0]["label"]
        top1_conf = top3[0]["confidence"]
        top2_conf = top3[1]["confidence"]
        margin = top1_conf - top2_conf

        # Only refine when uncertain — clear wins don't need it.
        if margin >= 0.25:
            return initial

        # Don't refine if the full-vocabulary confidence is too low — the
        # closed-world softmax will concentrate probability on whichever group
        # label best matches even for completely unrelated crops, producing
        # spuriously high fine-grained confidence.
        if top1_conf < 0.35:
            return initial

        group_key = self._LABEL_TO_GROUP.get(top1_label)
        if group_key is None:
            # Also check top-2 label's group.
            group_key = self._LABEL_TO_GROUP.get(top3[1]["label"])
        if group_key is None:
            return initial

        fine_labels = self._CONFUSABLE_GROUPS[group_key]

        # Build text features for the fine-grained label set (prompt ensembling).
        assert self.clip_processor is not None and self.clip_model is not None
        use_ensemble = os.getenv("CLIP_PROMPT_ENSEMBLE", "1").strip() in {"1", "true", "yes"}
        templates = _CLIP_PROMPT_TEMPLATES if use_ensemble else [self.clip_prompt_template]
        fine_feats: list[torch.Tensor] = []
        with torch.inference_mode():
            for tmpl in templates:
                t_texts = [tmpl.format(lbl) for lbl in fine_labels]
                tok = self.clip_processor(text=t_texts, return_tensors="pt", padding=True)
                tok = {k: v.to(self.device) for k, v in tok.items()}
                t_out = self.clip_model.text_model(**tok)
                t_feat = self.clip_model.text_projection(t_out.pooler_output)
                t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
                fine_feats.append(t_feat)
            stacked = torch.stack(fine_feats, dim=0).mean(dim=0)
            txt_feat = stacked / stacked.norm(dim=-1, keepdim=True)

            img_feat = self._image_to_features(image)
            scale = self.clip_model.logit_scale.exp()
            fine_logits = scale * (img_feat @ txt_feat.T)
            fine_probs = F.softmax(fine_logits[0], dim=-1)
            top_prob, top_idx = torch.topk(fine_probs, min(self.topk, fine_probs.shape[0]))

        fine_topk = [
            {"label": self.normalize_label(fine_labels[int(i)]),
             "confidence": float(p)}
            for p, i in zip(top_prob, top_idx)
        ]

        # Accept the fine-grained label when it dominates the closed-world set.
        # IMPORTANT: report the ORIGINAL full-vocabulary confidence, NOT the
        # closed-world softmax value.  Closed-world softmax always concentrates
        # mass on one label and produces spuriously high values (e.g. 99 %) even
        # for unrelated crops.  Keeping the initial confidence preserves honest
        # calibration while still benefiting from the narrower label set for
        # label selection and margin computation.
        if fine_topk[0]["confidence"] >= top1_conf * 0.85:
            # top-1 keeps the initial full-vocabulary confidence so downstream
            # _pick_label sees an honest calibrated score.  top-2+ use the actual
            # fine-grained probabilities so the margin is real, not manufactured.
            refined_top3 = [{"label": fine_topk[0]["label"], "confidence": top1_conf}]
            refined_top3 += fine_topk[1:]   # actual probabilities, not fabricated 0.0
            return {
                "food_name": fine_topk[0]["label"],
                "confidence": top1_conf,
                "top3": refined_top3,
            }
        return initial

    def _is_food_crop(self, crop: Image.Image) -> tuple[bool, float]:
        """Binary food/non-food gate using CLIP anchor embeddings.

        Returns (is_food, margin) where margin = food_max_similarity - nonfood_max_similarity.
        A positive margin means the crop looks more like food than non-food.
        Disable via env FOOD_GATE_ENABLED=0; tune via FOOD_GATE_MARGIN (default 0.0).
        """
        if (
            os.getenv("FOOD_GATE_ENABLED", "1").strip() not in {"1", "true", "yes"}
            or self.backend != "clip"
            or not hasattr(self, "_food_anchor_feat")
        ):
            return True, 1.0
        img_feat = self._image_to_features(crop)  # (1, D)
        with torch.inference_mode():
            # Use mean similarity across all anchors (not max) for a more robust
            # gate — a single high-scoring non-food anchor can't veto a food crop.
            food_mean    = float((img_feat @ self._food_anchor_feat.T)[0].mean())
            nonfood_mean = float((img_feat @ self._nonfood_anchor_feat.T)[0].mean())
        margin    = food_mean - nonfood_mean
        threshold = float(os.getenv("FOOD_GATE_MARGIN", "0.0"))
        return margin >= threshold, margin

    def classify(self, image: Image.Image | str, bbox: Sequence[Any] | None = None) -> Dict:
        """
        Classify food in image or cropped region.

        Returns:
            dict with food_name, confidence, top3.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image or file path.")

        image = image.convert("RGB")
        image = self._safe_crop(image, bbox)

        # Food/non-food gate — short-circuit before label classification.
        if self.backend == "clip":
            is_food, gate_margin = self._is_food_crop(image)
            if not is_food:
                return {
                    "food_name": "__nonfood__", "confidence": 0.0,
                    "top3": [], "gate_rejected": True, "gate_margin": gate_margin,
                }

        if self.backend == "clip":
            try:
                result = self._classify_clip(image)
                # Hierarchical re-pass: if top-1 falls in a confusable group
                # and margin is slim, narrow the label space and re-classify.
                result = self._hierarchical_refine(image, result)
                return result
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    self._switch_clip_to_cpu()
                    return self._classify_clip(image)
                raise

        return self._classify_vit(image)
