"""
Download pre-trained model weights from the GitHub release.

Usage:
    python scripts/download_weights.py

Downloads:
    weights/yolo.pt                          (~84 MB)  — YOLOv8l fine-tuned on 205-class food dataset
    weights/efficientnet_food101/best.pth    (~451 MB) — EfficientNetV2-L trained on Food-101 (90% val acc)
    weights/efficientnet_food101/labels.txt  (1 KB)    — 101-class label list

FastSAM and Depth Anything V2 weights must be downloaded separately (see README).
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Edit these URLs after creating your GitHub release
# ---------------------------------------------------------------------------
# Go to: https://github.com/<you>/<repo>/releases/new
# Upload: yolo.pt, best.pth, labels.txt
# Then paste the asset download links below.
WEIGHTS = {
    "weights/yolo.pt": (
        "https://github.com/maazxo1/CalCore-Pipeline/releases/download/v1.0/yolo.pt",
        84,
    ),
    "weights/efficientnet_food101/best.pth": (
        "https://github.com/maazxo1/CalCore-Pipeline/releases/download/v1.0/efficientnet_best.pth",
        451,
    ),
    "weights/efficientnet_food101/labels.txt": (
        "https://github.com/maazxo1/CalCore-Pipeline/releases/download/v1.0/labels.txt",
        0,
    ),
}

ROOT = Path(__file__).resolve().parent.parent


def _progress(downloaded: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, downloaded * block_size * 100 // total)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def download(name: str, url: str, expected_mb: int) -> None:
    dest = ROOT / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size // (1024 * 1024)
        if expected_mb == 0 or size_mb >= expected_mb * 0.9:
            print(f"  ✓ {name} already present — skipping")
            return
        print(f"  ⚠ {name} exists but looks incomplete ({size_mb} MB) — re-downloading")

    if "YOUR_USERNAME" in url:
        print(f"\n  ✗ URL not configured for {name}")
        print("    → Open scripts/download_weights.py and set the GitHub release URLs.")
        return

    print(f"\n  ↓ {name}  (~{expected_mb} MB)")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress bar
    except Exception as e:
        print(f"\n  ✗ Failed to download {name}: {e}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("  Food Pipeline — weight downloader")
    print("=" * 60)

    for name, (url, mb) in WEIGHTS.items():
        download(name, url, mb)

    print("\n✅ Done.")
    print("\nStill needed (download manually — see README):")
    print("  • weights/FastSAM.pt")
    print("  • weights/depth_anything_v2_metric_hypersim_vitl.pth")
    print("  • weights/depth_anything_v2_large.pth  (optional fallback)")


if __name__ == "__main__":
    main()
