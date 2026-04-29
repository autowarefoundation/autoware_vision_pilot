from __future__ import annotations

import math
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.load_data_auto_drive import CURV_SCALE  # noqa: E402
from Models.model_components.autodrive.autodrive_network import AutoDrive  # noqa: E402

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_NET_W, _NET_H = 1024, 512
_D_MAX_M = 150.0
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHEEL_PX = 92
_WHEEL_DIR_DEFAULT = _REPO_ROOT / "VisionPilot" / "production_release" / "images"


def _preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((_NET_W, _NET_H), Image.BILINEAR)
    t = TF.to_tensor(pil)
    return TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)


def _curvature_to_steer_deg(curv_1pm: float, wheelbase_m: float, steer_ratio: float) -> float:
    return math.degrees(math.atan(curv_1pm * wheelbase_m) * steer_ratio)


def _load_model(checkpoint: Path, device: torch.device) -> AutoDrive:
    model = AutoDrive().to(device)
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _text(img: np.ndarray, msg: str, x: int, y: int, scale=0.66, color=(255, 255, 255), thickness=1) -> None:
    cv2.putText(img, msg, (x + 1, y + 1), _FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, msg, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _load_wheel(path: Path, size_px: int) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return cv2.resize(img, (size_px, size_px), interpolation=cv2.INTER_LINEAR)


def _rotate_rgba(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _paste_rgba(base: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    oh, ow = overlay.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, base.shape[1]), min(y + oh, base.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    src = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    dst = base[y1:y2, x1:x2].astype(np.float32)
    base[y1:y2, x1:x2] = np.clip(src[:, :, :3].astype(np.float32) * alpha + dst * (1.0 - alpha), 0, 255).astype(np.uint8)


def _overlay_wheels(
    frame: np.ndarray,
    pred_steer: float,
    wheel_green: np.ndarray | None,
    wheel_white: np.ndarray | None,
) -> None:
    if wheel_green is None and wheel_white is None:
        return
    w = _WHEEL_PX
    gap = 6
    right = frame.shape[1] - 10
    y = 10
    x_pred = right - w
    x_ref = x_pred - gap - w
    ov = frame.copy()
    cv2.rectangle(ov, (x_ref - 8, y - 8), (right + 6, y + w + 28), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)
    if wheel_green is not None:
        _paste_rgba(frame, _rotate_rgba(wheel_green, 0.0), x_ref, y)
    if wheel_white is not None:
        _paste_rgba(frame, _rotate_rgba(wheel_white, pred_steer), x_pred, y)
    _text(frame, "Ref:0.0", x_ref + 8, y + w + 18, scale=0.44, color=(120, 240, 120), thickness=1)
    _text(frame, f"Pred:{pred_steer:+.1f}", x_pred + 2, y + w + 18, scale=0.44, color=(230, 230, 230), thickness=1)


def _draw_overlay(frame: np.ndarray, idx: int, dist_m: float, curv_1pm: float, steer_deg: float, flag_prob: float, flag_cls: int) -> np.ndarray:
    out = frame.copy()
    card_w = min(840, max(520, int(out.shape[1] * 0.58)))
    cv2.rectangle(out, (18, 18), (18 + card_w, 220), (20, 20, 20), -1)
    cv2.rectangle(out, (18, 18), (18 + card_w, 220), (120, 120, 120), 2)
    lines = [
        f"AutoDrive | Frame {idx}",
        f"Distance (m): {dist_m:.2f}",
        f"Curvature (1/m): {curv_1pm:+.5f}",
        f"Steering (deg): {steer_deg:+.2f}",
        f"CIPO Flag: {flag_cls}  (prob={flag_prob:.3f})",
    ]
    y0 = 52
    for i, txt in enumerate(lines):
        _text(out, txt, 36, y0 + i * 34, scale=0.76, color=(255, 255, 255), thickness=1)
    return out


def main():
    p = ArgumentParser(description="AutoDrive video visualization (original resolution)")
    p.add_argument("--checkpoint", required=True, help="AutoDrive .pth checkpoint")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output video path (.avi/.mp4)")
    p.add_argument("--start", type=int, default=0, help="Start frame index")
    p.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0=all)")
    p.add_argument("--fps", type=float, default=0.0, help="Output FPS (0=use input)")
    p.add_argument("--flag-threshold", type=float, default=0.65, help="CIPO flag threshold")
    p.add_argument("--wheelbase-m", type=float, default=2.984, help="Wheelbase (m)")
    p.add_argument("--steer-ratio", type=float, default=16.8, help="Steering column ratio")
    p.add_argument("--wheel-dir", type=str, default=str(_WHEEL_DIR_DEFAULT), help="Directory containing wheel_white.png and wheel_green.png")
    p.add_argument("--vis", action="store_true", default=False, help="Show live preview")
    args = p.parse_args()

    cap = cv2.VideoCapture(str(Path(args.video).expanduser().resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = float(cap.get(cv2.CAP_PROP_FPS))
    out_fps = args.fps if args.fps > 0 else (in_fps if in_fps > 0 else 20.0)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), out_fps, (in_w, in_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {out_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.checkpoint).expanduser().resolve(), device)
    wheel_dir = Path(args.wheel_dir).expanduser().resolve()
    wheel_white = _load_wheel(wheel_dir / "wheel_white.png", _WHEEL_PX)
    wheel_green = _load_wheel(wheel_dir / "wheel_green.png", _WHEEL_PX)
    if wheel_white is None:
        print(f"WARNING: wheel asset missing: {wheel_dir / 'wheel_white.png'}")
    if wheel_green is None:
        print(f"WARNING: wheel asset missing: {wheel_dir / 'wheel_green.png'}")

    ok, prev = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("No frames in input video")

    frame_idx, processed = 0, 0
    while True:
        ok, curr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx < args.start:
            prev = curr
            continue
        if args.max_frames > 0 and processed >= args.max_frames:
            break

        t_prev = _preprocess(prev).unsqueeze(0).to(device)
        t_curr = _preprocess(curr).unsqueeze(0).to(device)
        with torch.no_grad():
            d_pred, curv_pred, flag_logit = model(t_prev, t_curr)

        pred_d_norm = float(d_pred.squeeze().cpu())
        pred_dist_m = _D_MAX_M * (1.0 - max(0.0, min(1.0, pred_d_norm)))
        pred_curv_1pm = float(curv_pred.squeeze().cpu()) * CURV_SCALE
        pred_flag_prob = float(torch.sigmoid(flag_logit.squeeze()).cpu())
        pred_flag_cls = 1 if pred_flag_prob >= args.flag_threshold else 0
        pred_steer_deg = _curvature_to_steer_deg(pred_curv_1pm, args.wheelbase_m, args.steer_ratio)

        vis = _draw_overlay(curr, frame_idx, pred_dist_m, pred_curv_1pm, pred_steer_deg, pred_flag_prob, pred_flag_cls)
        _overlay_wheels(vis, pred_steer_deg, wheel_green, wheel_white)
        writer.write(vis)

        if args.vis:
            cv2.imshow("AutoDrive", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed += 1
        prev = curr

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done — {processed} frames saved to: {out_path}")


if __name__ == "__main__":
    main()
