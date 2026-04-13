"""
AutoDrive ZOD dataset loader.

Labels:  {zod_root}/labels/{seq}/*.json  (one per frame, ISO-timestamp filename)
Images:  {zod_root}/images_blur_*/sequences/{seq}/camera_front_blur/

Image preprocessing — identical to AutoSpeed inference pipeline:
    ZOD camera has ~120° HFOV.  We center-crop to 50° HFOV then resize to
    1024×512 with bilinear interpolation.  This is the exact same function
    used in run_cipo_radar.py → center_crop_50deg_resize.

    Sequence:
        1. Open raw PIL image (full res, ~120° FOV)
        2. Center-crop to 50° HFOV  (crop_w = img_w * 50 / hfov_deg, 2:1 ratio)
        3. Resize to 1024×512 with PIL BILINEAR
        4. Convert to numpy HWC uint8
        5. Training only: horizontal flip (negates curvature) + colour/noise augmentation
           applied identically to both frames via albumentations ReplayCompose
        6. Normalise and convert to CHW float32 tensor (ImageNet stats)

Sequential pairs:
    (T-1, T) within each sequence only — no cross-sequence pairing.

Split:  85 / 10 / 5 at sequence level to avoid temporal leakage.

Distance GT:
    d_norm = (150 - min(d, 150)) / 150  →  ∈ [0, 1]
    dist_mask=True  only when cipo_detected=True AND distance is valid.
    dist_mask=False → distance loss is zero for that sample.

Curvature normalisation:
    Raw ZOD curvature spans ≈ ±0.21 (1/m) across the full dataset.
    CURV_SCALE = 0.21 maps that range onto [-1, 1], matching the Tanh output.
    The dataset returns curvature / CURV_SCALE so the L1 loss operates on the
    same ±1 scale as the head output.  To convert model predictions back to
    physical (1/m) units: pred_1_per_m = pred_normalised * CURV_SCALE.
"""

import json
import random
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from Models.data_parsing.AutoDrive.zod.zod_utils import (
    get_images_blur_dir,
    get_calibration_path,
)

# ── Network input size (must match AutoSpeed training resolution) ──────────
_NET_W, _NET_H   = 1024, 512
_TARGET_FOV      = 50.0   # degrees — same as AutoSpeed/run_cipo_radar
_ZOD_HFOV_DEG   = 120.0  # fallback; overridden by calibration file
_D_MAX           = 150.0  # metres

# ── Curvature normalisation ────────────────────────────────────────────────
# Empirical max |curvature| across the full ZOD dataset (296 k frames):
#   min = -0.2099 (1/m)   max = +0.2090 (1/m)
# Dividing by CURV_SCALE maps GT onto [-1, 1] — same scale as Tanh output.
# Callers that need physical units: pred_1_per_m = pred_norm * CURV_SCALE
CURV_SCALE: float = 0.21

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Colour / noise augmentation — applied identically to both frames ───────
_COLOUR_AUG = A.ReplayCompose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
    A.GaussNoise(noise_scale_factor=0.2, p=0.3),
    A.ISONoise(color_shift=(0.05, 0.2), intensity=(0.1, 0.3), p=0.2),
    A.ToGray(num_output_channels=3, method='weighted_average', p=0.05),
])


# ── Image preprocessing ────────────────────────────────────────────────────

def _center_crop_50deg_resize(img: Image.Image, hfov_deg: float) -> np.ndarray:
    """
    Center-crop to 50° HFOV then resize to 1024×512.

    Exact same logic as center_crop_50deg_resize() in run_cipo_radar.py:
        crop_w = round(img_w * 50 / hfov_deg)   ← proportion of 50° from full FOV
        crop_h = crop_w // 2                      ← 2:1 aspect ratio
        Crop centered both horizontally and vertically.
        Resize with PIL BILINEAR to 1024×512.

    Uses actual image dims (img.size), not calibration W/H (may differ).
    Returns: numpy HWC uint8 array of shape (512, 1024, 3).
    """
    img_w, img_h  = img.size
    orig_crop_w   = int(round(img_w * _TARGET_FOV / hfov_deg))
    orig_crop_h   = orig_crop_w // 2                            # 2:1 ratio
    crop_x        = (img_w - orig_crop_w) // 2
    crop_y        = (img_h - orig_crop_h) // 2                 # centered vertically
    cropped       = img.crop((crop_x, crop_y,
                               crop_x + orig_crop_w,
                               crop_y + orig_crop_h))
    model_img     = cropped.resize((_NET_W, _NET_H), Image.BILINEAR)
    return np.array(model_img)


def _read_hfov_deg(zod_root: Path, seq: str) -> float:
    """Read horizontal FOV (degrees) from the sequence calibration file."""
    calib_path = get_calibration_path(zod_root, seq)
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)["FC"]
        return float(calib["field_of_view"][0])
    return _ZOD_HFOV_DEG  # fallback


def _norm_distance(d_metres: float) -> float:
    return (_D_MAX - min(d_metres, _D_MAX)) / _D_MAX


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    img = TF.to_tensor(img_np)
    img = TF.normalize(img, _IMAGENET_MEAN, _IMAGENET_STD)
    return img


# ── Augmentations ──────────────────────────────────────────────────────────

def _augment_pair(img_prev: np.ndarray, img_curr: np.ndarray,
                  curvature: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Training augmentations applied to a (prev, curr) image pair.

    Horizontal flip (p=0.5):
        Both frames flipped left-right simultaneously.
        Curvature sign negated  (right curve → left curve after flip).
        Distance and flag are unchanged (symmetric).

    Colour / noise:
        Random parameters drawn once, replayed identically on both frames
        via albumentations ReplayCompose to preserve temporal consistency.
    """
    if random.random() < 0.5:
        img_prev  = np.ascontiguousarray(img_prev[:, ::-1, :])
        img_curr  = np.ascontiguousarray(img_curr[:, ::-1, :])
        curvature = -curvature

    result   = _COLOUR_AUG(image=img_prev)
    img_prev = result["image"]
    img_curr = A.ReplayCompose.replay(result["replay"], image=img_curr)["image"]

    return img_prev, img_curr, curvature


# ── Dataset ────────────────────────────────────────────────────────────────

class AutoDriveDataset(Dataset):
    """
    Each __getitem__ returns:
        img_prev  : (3, 512, 1024) float tensor  — ImageNet-normalised
        img_curr  : (3, 512, 1024) float tensor
        d_norm    : scalar float ∈ [0, 1]
        curvature : scalar float ∈ [-1, 1]  — normalised by CURV_SCALE (0.21)
                    (sign flipped on horizontal flip; convert to 1/m: × CURV_SCALE)
        flag      : scalar float {0.0, 1.0}  — 1 = CIPO present
        dist_mask : bool  — True = distance loss active for this sample
    """

    def __init__(self, zod_root: str | Path, sequences: list[str],
                 is_train: bool = True):
        self.is_train = is_train
        self.pairs: list[tuple] = []

        zod_root = Path(zod_root)

        # Read hfov_deg from the first sequence that has a calibration file.
        # All ZOD sequences share the same camera, so this is consistent.
        self.hfov_deg = _ZOD_HFOV_DEG
        for seq in sequences:
            calib_path = get_calibration_path(zod_root, seq)
            if calib_path.exists():
                self.hfov_deg = _read_hfov_deg(zod_root, seq)
                break
        print(f"  Camera HFOV: {self.hfov_deg:.1f}°  →  50° crop  →  {_NET_W}×{_NET_H}")

        for seq in sequences:
            label_dir = zod_root / "labels" / seq
            if not label_dir.exists():
                continue

            label_files = sorted(label_dir.glob("*.json"))
            if len(label_files) < 2:
                continue

            img_dir = get_images_blur_dir(zod_root, seq)
            records = []
            for lf in label_files:
                with open(lf) as fh:
                    rec = json.load(fh)
                img_path = img_dir / rec["image"]
                if img_path.exists():
                    records.append((str(img_path), rec))

            for i in range(1, len(records)):
                path_prev, _        = records[i - 1]
                path_curr, lbl_curr = records[i]

                cipo      = bool(lbl_curr.get("cipo_detected", False))
                raw_dist  = lbl_curr.get("distance_to_in_path_object")
                curvature = float(lbl_curr.get("curvature") or 0.0)

                if cipo and raw_dist is not None:
                    d_norm    = _norm_distance(float(raw_dist))
                    dist_mask = True
                else:
                    d_norm    = 0.0
                    dist_mask = False

                flag = 1.0 if cipo else 0.0
                self.pairs.append((path_prev, path_curr,
                                   d_norm, curvature, flag, dist_mask))

        print(f"AutoDriveDataset ({'train' if is_train else 'val/test'}): "
              f"{len(self.pairs):,} pairs from {len(sequences)} sequences.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        path_prev, path_curr, d_norm, curvature, flag, dist_mask = self.pairs[idx]

        # 1. Load raw PIL images
        img_prev = Image.open(path_prev).convert("RGB")
        img_curr = Image.open(path_curr).convert("RGB")

        # 2. Center-crop to 50° HFOV then resize to 1024×512
        #    Exact same method as center_crop_50deg_resize() in run_cipo_radar.py
        img_prev = _center_crop_50deg_resize(img_prev, self.hfov_deg)
        img_curr = _center_crop_50deg_resize(img_curr, self.hfov_deg)

        # 3. Training augmentations (flip + colour/noise)
        if self.is_train:
            img_prev, img_curr, curvature = _augment_pair(img_prev, img_curr, curvature)

        # 4. Normalise and convert to tensor
        # Curvature: divide by CURV_SCALE so GT ∈ [-1, 1], matching Tanh output.
        curv_norm = curvature / CURV_SCALE
        return {
            "img_prev":  _to_tensor(img_prev),
            "img_curr":  _to_tensor(img_curr),
            "d_norm":    torch.tensor(d_norm,    dtype=torch.float32),
            "curvature": torch.tensor(curv_norm, dtype=torch.float32),
            "flag":      torch.tensor(flag,      dtype=torch.float32),
            "dist_mask": torch.tensor(dist_mask, dtype=torch.bool),
        }


# ── Splitter ───────────────────────────────────────────────────────────────

class LoadDataAutoDrive:
    """
    Splits all ZOD sequences 85 / 10 / 5 at sequence level to avoid
    temporal leakage.

        data = LoadDataAutoDrive("/path/to/zod")
        data.train / data.val / data.test  →  AutoDriveDataset
    """

    TRAIN_FRAC = 0.85
    VAL_FRAC   = 0.10

    def __init__(self, zod_root: str | Path):
        zod_root   = Path(zod_root)
        labels_dir = zod_root / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        all_seqs = sorted([d.name for d in labels_dir.iterdir() if d.is_dir()])
        if not all_seqs:
            raise FileNotFoundError(f"No sequence folders found under {labels_dir}")

        n       = len(all_seqs)
        n_train = max(1, round(n * self.TRAIN_FRAC))
        n_val   = max(1, round(n * self.VAL_FRAC))

        train_seqs = all_seqs[:n_train]
        val_seqs   = all_seqs[n_train : n_train + n_val]
        test_seqs  = all_seqs[n_train + n_val :]

        print(f"Sequences — train: {len(train_seqs)}  val: {len(val_seqs)}  test: {len(test_seqs)}")

        self.train = AutoDriveDataset(zod_root, train_seqs, is_train=True)
        self.val   = AutoDriveDataset(zod_root, val_seqs,   is_train=False)
        self.test  = AutoDriveDataset(zod_root, test_seqs,  is_train=False)
