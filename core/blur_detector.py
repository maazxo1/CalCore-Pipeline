"""
FFT-based Blur Detection
========================
Adapted from VolETA-MetaFood (CVPR 2024 Food Portion Challenge Winner).

Principle:
    Sharp images have strong high-frequency components (edges, texture).
    Blurry images have most energy concentrated in low frequencies.
    Algorithm:
        1. Compute 2-D FFT of the grayscale image.
        2. Zero out the central low-frequency region.
        3. Inverse FFT → high-frequency signal only.
        4. Mean log-magnitude of the residual = sharpness score.
        5. score <= threshold  →  image is too blurry to process.

The standard width (500 px) normalization ensures the threshold is
image-size independent — same approach used in VolETA's pipeline.
"""

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Public constants (can be overridden per call)
# --------------------------------------------------------------------------
DEFAULT_THRESHOLD = 10.0   # VolETA's value for food images; lower = stricter
_STANDARD_WIDTH   = 500    # Resize to this width before analysis
_CENTER_SIZE      = 60     # Half-size of the low-freq region to zero out


def detect_blur(
    image_bgr: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple:
    """Detect whether an image is too blurry to process reliably.

    Adapted from VolETA-MetaFood (CVPR 2024).

    Args:
        image_bgr : BGR image array (H, W, 3) as returned by cv2.imread /
                    cv2.imdecode.  Grayscale (H, W) also accepted.
        threshold : Sharpness score below which the image is flagged blurry.
                    Default 10.0 (VolETA's calibrated value for food photos).

    Returns:
        Tuple (score: float, is_blurry: bool).
        score is the mean high-frequency log-magnitude; higher = sharper.
        Typical values: sharp ≈ 20–40, slightly soft ≈ 12–18, blurry < 10.
    """
    # ── Convert to grayscale ──────────────────────────────────────────────
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    # ── Resize to standard width (makes threshold image-size independent) ─
    h, w = gray.shape
    new_h = max(1, int(h * _STANDARD_WIDTH / w))
    gray = cv2.resize(gray, (_STANDARD_WIDTH, new_h))

    H, W = gray.shape
    cx, cy = W // 2, H // 2

    # ── Adaptive centre block — capped to ¼ of the smaller dimension ─────
    # _CENTER_SIZE=60 assumes a roughly square image.  Wide-but-short crops
    # (bacon strips, pancakes, pizza slices) resize to e.g. 500×70 px, so
    # cy=35 and cy-60 goes negative → numpy clamps it → the entire height
    # gets zeroed out → bogus near-zero score → sharp food falsely "blurry".
    center = min(_CENTER_SIZE, H // 4, W // 4)
    if center < 1:
        center = 1

    # ── FFT → shift zero-frequency to centre ─────────────────────────────
    fft       = np.fft.fft2(gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)

    # ── Zero out low-frequency (DC + smooth) centre block ────────────────
    fft_shift[cy - center: cy + center,
              cx - center: cx + center] = 0

    # ── Inverse transform → high-frequency residual ───────────────────────
    recon     = np.fft.ifft2(np.fft.ifftshift(fft_shift))

    # ── Mean log-magnitude = sharpness score ─────────────────────────────
    magnitude = 20.0 * np.log(np.abs(recon) + 1e-10)   # +eps avoids log(0)
    score     = float(np.mean(magnitude))

    return score, score <= threshold
