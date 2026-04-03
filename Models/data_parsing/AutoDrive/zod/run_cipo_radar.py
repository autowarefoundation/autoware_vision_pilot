#!/usr/bin/env python3
"""
Run AutoSpeed (letterboxed inference, auto_speed_infer.AutoSpeedNetworkInfer) on ZOD images,
compute CIPO azimuth in camera frame, transform to radar frame, associate with nearest radar cluster.
Output: distance (m), speed (m/s) per image.
"""

import json
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add repo root for Models import
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image

try:
    from Models.inference.auto_speed_infer import AutoSpeedNetworkInfer
except ImportError:
    from inference.auto_speed_infer import AutoSpeedNetworkInfer

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None


_LAT_BUFFER_M = 0.5        # ±0.5m lateral buffer for CIPO-radar (Scenario 1) azimuth association
_LAT_BUFFER_RELAXED_M = 1.0  # fallback cone for CIPO when strict cone finds no cluster
_LAT_BUFFER_PATH_M = 1.0   # no-CIPO path search: ±1.0m lateral from curvature path
_MIN_ABS_SPEED_WORLD_MS = 0.5   # Scenario 3: |range_rate + ego_speed| > 0.5 → moving object
_MIN_ABS_RANGE_RATE_FALLBACK = 0.5  # fallback when ego_speed not available
_MAX_RANGE_M = 200.0        # maximum radar search range (m) - radar useful up to ~200m

# Temporal association: search up to N frames forward/back.
# - Pass 1 (CIPO, cluster miss): track_from_prev only looks *backward* in `results` (online; future frames not built yet).
# - Pass 2 (CIPO bbox, no range): lookahead + lookbehind N for same-object D/V extrapolation.
# - Pass 3 (no CIPO camera): lookahead + lookbehind N for path-radar neighbor fill (moving target).
TEMPORAL_NEIGHBOR_FRAMES = 10

# Volvo XC90 (ZOD vehicle) steering geometry - same as step1_timestamp_association.py
_STEERING_COLUMN_RATIO = 16.8  # steering wheel deg / tyre deg

_ZOD_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_ZOD_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_ZOD_SCRIPT_DIR))
from zod_utils import (
    default_autospeed_checkpoint,
    get_images_blur_dir,
    get_calibration_path,
    sequence_output_dir,
)


def parse_image_timestamp(fname: Path) -> int:
    stem = fname.stem
    parts = stem.split("_")
    ts_str = "_".join(parts[2:])
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1e9)




def radar_spherical_to_cartesian(pts):
    """Radar: X forward, Y left, Z up. ZOD angles in radians."""
    az = pts["azimuth_angle"].astype(np.float64)
    el = pts["elevation_angle"].astype(np.float64)
    rg = pts["radar_range"].astype(np.float64)
    x = rg * np.cos(el) * np.cos(az)
    y = rg * np.cos(el) * np.sin(az)
    z = rg * np.sin(el)
    return x, y, z


def pixel_to_h_angle_deg(u: float, W: float, H: float, hfov_deg: float) -> float:
    """
    Horizontal angle (deg) from optical axis.
    H_angle = ((u - W/2) / (W/2)) * (HFOV/2)
    """
    return ((u - W / 2) / (W / 2)) * (hfov_deg / 2)


_MODEL_W, _MODEL_H = 1024, 512  # AutoSpeedNetworkInfer train size


def center_crop_50deg_resize(img: "Image.Image", W: int, H: int, hfov_deg: float, target_fov: float = 50.0):
    """50° HFOV crop: width = img_w*target_fov/hfov_deg, height = width/2 (2:1), centered; resize to 1024×512."""
    img_w, img_h = img.size  # use actual image dims, not calibration (may differ)
    orig_crop_w = int(round(img_w * target_fov / hfov_deg))   # ≈ 1603 for ZOD
    orig_crop_h = orig_crop_w // 2                             # 2:1 → ≈ 801
    crop_x = (img_w - orig_crop_w) // 2
    crop_y = (img_h - orig_crop_h) // 2                       # center vertically
    cropped = img.crop((crop_x, crop_y, crop_x + orig_crop_w, crop_y + orig_crop_h))
    model_img = cropped.resize((_MODEL_W, _MODEL_H), Image.BILINEAR)
    crop_info = {
        "crop_x": crop_x,
        "crop_y": crop_y,
        "orig_crop_w": orig_crop_w,
        "orig_crop_h": orig_crop_h,
        "model_w": _MODEL_W,
        "model_h": _MODEL_H,
        "fov_deg": target_fov,
    }
    return model_img, crop_info


def pixel_to_h_angle_deg_50(u: float, crop_info: dict) -> float:
    """
    Same convention as reference `pixel_to_h_angle_deg_50`: angle from optical axis in the 50° window.
    Here `u IS THE HORIZONTAL CENTER OF THE BBOX` in **model input pixels** (0..model_w, usually 1024x512).
    Map to full-frame horizontal position in that window, then linear angle in the 50° span.
    """
    u_full = crop_info["crop_x"] + (u / crop_info["model_w"]) * crop_info["orig_crop_w"]
    cw = crop_info["orig_crop_w"]
    fov = crop_info["fov_deg"]
    return ((u_full - crop_info["crop_x"] - cw / 2) / (cw / 2)) * (fov / 2)


def bbox_crop_to_full(x1, y1, x2, y2, crop_info):
    """
    Map bbox corners from model space (1024×512) back to full-image coordinates.
    x: scale by (orig_crop_w / model_w) + crop_x offset.
    y: scale by (orig_crop_h / model_h) + crop_y offset.
    """
    sx = crop_info["orig_crop_w"] / crop_info["model_w"]
    sy = crop_info["orig_crop_h"] / crop_info["model_h"]
    cx, cy = crop_info["crop_x"], crop_info["crop_y"]
    return float(x1) * sx + cx, float(y1) * sy + cy, float(x2) * sx + cx, float(y2) * sy + cy


def cam_dir_to_radar_azimuth(h_angle_deg: float, cam_ext: np.ndarray, radar_ext: np.ndarray) -> float:
    """
    Transform camera horizontal angle to radar azimuth (radians).
    Camera: X right, Y down, Z forward. H_angle from optical axis (Z).
    dir_cam = (sin(h), 0, cos(h)) in camera frame.
    Transform via extrinsics: dir_radar = R_radar^T @ R_cam @ dir_cam
    azimuth_radar = atan2(dir_radar[1], dir_radar[0])
    """
    h_rad = np.deg2rad(h_angle_deg)
    dir_cam = np.array([np.sin(h_rad), 0.0, np.cos(h_rad)])
    R_cam = np.array(cam_ext)[:3, :3]
    R_radar = np.array(radar_ext)[:3, :3]
    dir_world = R_cam @ dir_cam
    dir_radar = R_radar.T @ dir_world
    return float(np.arctan2(dir_radar[1], dir_radar[0]))


def _polar_vel_dist(a, b, range_scale=4.0, lat_buffer=0.5, vel_scale=1.5):
    """Polar+velocity distance: range ~4m, lateral ~0.5m, velocity ~1.5 m/s. a,b are (range, azimuth, range_rate)."""
    dr = abs(a[0] - b[0])
    r_avg = (a[0] + b[0]) / 2
    daz = abs(np.angle(np.exp(1j * (a[1] - b[1]))))
    d_lateral = r_avg * abs(np.sin(daz)) if r_avg > 0 else 0.0
    dv = abs(a[2] - b[2])
    return np.sqrt((dr / range_scale) ** 2 + (d_lateral / lat_buffer) ** 2 + (dv / vel_scale) ** 2)


_MIN_ABS_RR_SINGLE_PT = 0.5  # unclustered point is "moving" if |range_rate| > 0.5 m/s

def get_radar_clusters(radar_data, ts_ns: int, z_min=-0.5, z_max=1.0, range_scale=4.0, lat_buffer=0.5, vel_scale=1.5, min_samples=2, max_range_m=_MAX_RANGE_M):
    """
    Filter z (-0.5 to 1m: ground to car roof) and range (≤max_range_m), cluster with DBSCAN.
    After DBSCAN, any unclustered point with |range_rate| > _MIN_ABS_RR_SINGLE_PT is promoted
    to its own single-point cluster (moving object missed by density clustering).
    """
    pts = radar_data[radar_data["timestamp"] == ts_ns]
    if len(pts) == 0:
        return []
    x, y, z = radar_spherical_to_cartesian(pts)
    rg_all = pts["radar_range"].astype(np.float64)
    mask = (z >= z_min) & (z <= z_max) & (rg_all <= max_range_m)
    pts_f = pts[mask]
    rg = pts_f["radar_range"].astype(np.float64)
    az = pts_f["azimuth_angle"].astype(np.float64)
    rr = pts_f["range_rate"].astype(np.float64)
    polar_vel = np.column_stack([rg, az, rr])
    if len(polar_vel) == 0:
        return []
    if DBSCAN is None:
        raise RuntimeError("DBSCAN unavailable (scikit-learn not installed). Install `scikit-learn`.")
    metric = lambda a, b: _polar_vel_dist(a, b, range_scale, lat_buffer, vel_scale)
    labels = DBSCAN(eps=1.0, min_samples=min_samples, metric=metric).fit(polar_vel).labels_
    clusters = []
    for lbl in set(labels):
        if lbl < 0:
            continue
        m = labels == lbl
        clusters.append({
            "azimuth": float(np.mean(pts_f["azimuth_angle"][m])),
            "range": float(np.mean(pts_f["radar_range"][m])),
            "range_rate": float(np.mean(pts_f["range_rate"][m])),
        })
    # Promote moving unclustered points to single-point clusters
    unclustered = labels < 0
    for i in np.where(unclustered)[0]:
        if abs(rr[i]) > _MIN_ABS_RR_SINGLE_PT:
            clusters.append({
                "azimuth": float(az[i]),
                "range": float(rg[i]),
                "range_rate": float(rr[i]),
            })
    return clusters


def find_nearest_cluster_lateral(clusters, azimuth_radar: float, lat_buffer_m: float = 0.5):
    """
    Filter clusters within ±lat_buffer_m lateral distance from CIPO ray.
    Perpendicular distance = r * |sin(az_cluster - az_cipo)| <= lat_buffer_m.
    Among those, pick the one with minimum range (closest along the azimuth ray).
    """
    if not clusters:
        return None
    in_cone = []
    for c in clusters:
        daz = abs(np.angle(np.exp(1j * (c["azimuth"] - azimuth_radar))))
        d_lateral = c["range"] * abs(np.sin(daz))
        if d_lateral <= lat_buffer_m:
            in_cone.append(c)
    if not in_cone:
        return None
    return min(in_cone, key=lambda c: c["range"])


def _path_azimuth_at_range(curvature_inv_m: float, range_m: float) -> float:
    """
    Azimuth (rad) of the curvature path at given range from ego.
    Circular arc: x=R*sin(θ), y=R*(1-cos(θ)); at range r, θ=2*arcsin(r/(2R)).
    az = atan2(y,x). Small-angle: az ≈ κ*r/2 (NOT κ*r).
    """
    k = curvature_inv_m
    if abs(k) < 1e-9:
        return 0.0
    R = 1.0 / k
    r = min(range_m, 2 * R - 1e-6)  # avoid arcsin > 1
    theta = 2 * np.arcsin(r / (2 * R))
    x = R * np.sin(theta)
    y = R * (1 - np.cos(theta))
    return float(np.arctan2(y, x))


def find_nearest_cluster_on_path(clusters, curvature_inv_m: float, lat_buffer_m: float = 0.5):
    """
    Cluster most on the path (smallest lateral deviation), not nearest by range.
    Pavement at 10m off-path < car at 20m on-path.
    """
    if not clusters:
        return None
    in_path = []
    for c in clusters:
        r, az = c["range"], c["azimuth"]
        az_path = _path_azimuth_at_range(curvature_inv_m, r)
        daz = abs(np.angle(np.exp(1j * (az - az_path))))
        d_lateral = r * abs(np.sin(daz))
        if d_lateral <= lat_buffer_m:
            in_path.append((c, d_lateral))
    if not in_path:
        return None
    # Pick cluster closest to path (min lateral deviation), then nearest by range as tiebreaker
    return min(in_path, key=lambda x: (x[1], x[0]["range"]))[0]


def find_cluster_on_path_direct(
    radar_data,
    ts_ns: int,
    curvature_inv_m: float,
    lat_buffer_m: float = 1.0,
    z_min: float = -0.5,
    z_max: float = 1.0,
    range_gap_m: float = 4.0,
    vel_gap_ms: float = 3.0,
    D_est: float = None,
    range_tol_m: float = 3.0,
    V_ref: float = None,
    vel_tol_ms: float = 2.0,
    min_pts: int = 2,
    min_abs_range_rate: float = None,
    min_abs_speed_world: float = None,
    ego_speed_ms: float = None,
    max_range_m: float = _MAX_RANGE_M,
):
    """
    No-CIPO path search on raw radar points (no DBSCAN, no FOV/azimuth constraint).
    Filters individual points within lat_buffer_m of the curvature path, groups by
    range+velocity proximity, returns best cluster (min lateral deviation).

    min_pts=2 (default, Pass 1): require ≥2 points so single noise/footpath returns are rejected.
    min_pts=1 (Pass 3): allow single point when a neighboring frame already confirmed the object.
    Scenario 3 moving filter: radar range_rate is relative to ego. Object speed (world) = range_rate + ego_speed.
    Static object: |range_rate + ego_speed| < threshold. Use min_abs_speed_world + ego_speed_ms when available.
    Fallback: min_abs_range_rate if ego_speed_ms is None.
    """
    pts = radar_data[radar_data["timestamp"] == ts_ns]
    if len(pts) == 0:
        return None
    x, y, z = radar_spherical_to_cartesian(pts)
    mask = (z >= z_min) & (z <= z_max)
    pts_f = pts[mask]
    if len(pts_f) == 0:
        return None
    rg = pts_f["radar_range"].astype(np.float64)
    az = pts_f["azimuth_angle"].astype(np.float64)
    rr = pts_f["range_rate"].astype(np.float64)

    # Filter to points within lat_buffer_m of curvature path and within max_range_m
    on_path = []
    for i in range(len(pts_f)):
        if rg[i] > max_range_m:
            continue
        az_path = _path_azimuth_at_range(curvature_inv_m, rg[i])
        daz = abs(np.angle(np.exp(1j * (az[i] - az_path))))
        d_lateral = rg[i] * abs(np.sin(daz))
        if d_lateral > lat_buffer_m:
            continue
        if D_est is not None and abs(rg[i] - D_est) > range_tol_m:
            continue
        if V_ref is not None and abs(rr[i] - V_ref) > vel_tol_ms:
            continue
        # Scenario 3: exclude static objects. Radar range_rate is relative; object_speed_world = range_rate + ego_speed
        if ego_speed_ms is not None and min_abs_speed_world is not None:
            if abs(rr[i] + ego_speed_ms) < min_abs_speed_world:
                continue  # static in world frame
        elif min_abs_range_rate is not None:
            if abs(rr[i]) < min_abs_range_rate:
                continue  # fallback when ego_speed not available
        on_path.append((i, float(rg[i]), float(az[i]), float(rr[i]), float(d_lateral)))

    if not on_path:
        return None

    # Sort by range, greedy-group nearby points
    on_path.sort(key=lambda p: p[1])
    groups = [[on_path[0]]]
    for pt in on_path[1:]:
        last = groups[-1][-1]
        if abs(pt[1] - last[1]) <= range_gap_m and abs(pt[3] - last[3]) <= vel_gap_ms:
            groups[-1].append(pt)
        else:
            groups.append([pt])

    # Pick group with ≥ min_pts, min lateral deviation from path, then min range
    best = None
    best_score = (float("inf"), float("inf"))
    for group in groups:
        if len(group) < min_pts:
            continue
        indices = np.array([p[0] for p in group])
        mean_dlat = float(np.mean([p[4] for p in group]))
        mean_range = float(np.mean([p[1] for p in group]))
        score = (mean_dlat, mean_range)
        if score < best_score:
            best_score = score
            best = {
                "range": mean_range,
                "azimuth": float(np.mean([p[2] for p in group])),
                "range_rate": float(np.mean([p[3] for p in group])),
                "indices": indices,
            }
    return best


def _rec_meta(rec: dict) -> dict:
    """Extract association metadata to embed in every result record."""
    steering = rec.get("steering_angle_rad", 0.0) or 0.0
    return {
        "image_timestamp_ns": rec.get("image_timestamp_ns"),
        "radar_timestamp_ns": rec.get("radar_timestamp_ns"),
        "curvature_inv_m": rec.get("curvature_inv_m"),
        "steering_angle_rad": rec.get("steering_angle_rad"),
        "tyre_angle_rad": round(float(steering) / _STEERING_COLUMN_RATIO, 7),
        "ego_speed_ms": rec.get("ego_speed_ms"),
    }


def _viz_fields_from_cluster(cluster, azimuth_rad_deg=None):
    """
    Compute visualization fields from cluster: bev_xy, speed_ms_adjusted.
    cluster: dict with range, range_rate, optionally azimuth (radians).
    azimuth_rad_deg: used when cluster has no azimuth (e.g. track_from_prev).
    Returns dict with bev_xy [x,y], speed_ms_adjusted.
    speed_ms_adjusted = range_rate * cos(azimuth) = longitudinal component of relative velocity.
    """
    if cluster is None:
        return {}
    r = cluster["range"]
    rr = cluster["range_rate"]
    az = cluster.get("azimuth")
    if az is None and azimuth_rad_deg is not None:
        az = np.deg2rad(azimuth_rad_deg)
    elif az is None:
        return {}
    az_rad = float(az)
    x_bev = r * np.cos(az_rad)
    y_bev = r * np.sin(az_rad)
    speed_adj = rr * np.cos(az_rad)
    return {
        "bev_xy": [round(x_bev, 2), round(y_bev, 2)],
        "speed_ms_adjusted": round(speed_adj, 2),
    }


def _pixel_point_from_bbox(bbox):
    """Extract bottom-center pixel [u, v] from bbox [x1, y1, x2, y2]."""
    if bbox is None or len(bbox) < 4:
        return None
    u = (bbox[0] + bbox[2]) / 2
    v = bbox[3]  # bottom edge (y2)
    return [round(float(u), 1), round(float(v), 1)]


def find_cipo_via_bbox(
    preds,
    curvature_inv_m,
    clusters,
    cam_ext,
    radar_ext,
    crop_info: dict,
    lat_buffer_m=0.5,
    path_buffer_m=1.0,
):
    """
    Scenario 2: no L1/L2 CIPO detected but other bounding boxes exist.
    For each bbox sorted by bottom-y (closest first), project to radar azimuth,
    find nearest radar cluster, verify it lies on the curvature path.
    Bbox coords are in 50°-crop space (inference output). crop_info used for angle mapping.
    Returns (cluster, az_radar_rad) for the first on-path bbox match, or (None, None).
    """
    if not preds or not clusters:
        return None, None
    sorted_preds = sorted(preds, key=lambda p: (p[1] + p[3]) / 2, reverse=True)
    for pred in sorted_preds:
        x1, y1, x2, y2, conf, cls = pred
        u = (x1 + x2) / 2
        h_angle_deg = pixel_to_h_angle_deg_50(u, crop_info)
        az_radar = cam_dir_to_radar_azimuth(h_angle_deg, cam_ext, radar_ext)
        cluster = find_nearest_cluster_lateral(clusters, az_radar, lat_buffer_m=lat_buffer_m)
        if cluster is None:
            continue
        az_path = _path_azimuth_at_range(curvature_inv_m, cluster["range"])
        daz = abs(np.angle(np.exp(1j * (cluster["azimuth"] - az_path))))
        d_lateral = cluster["range"] * abs(np.sin(daz))
        if d_lateral <= path_buffer_m:
            return cluster, az_radar
    return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="000479")
    parser.add_argument("--zod-root", type=str, default=None, required=True, help="ZOD dataset root (contains images_blur_* folders, radar_front/, infos/, vehicle_data/, etc.)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for cipo_radar.json (default: {zod_root}/output/{seq}/cipo_radar.json)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to AutoSpeed weights (default: {zod_root}/models/autospeed.pt)",
    )
    args = parser.parse_args()
    seq = args.sequence
    zod = Path(args.zod_root)
    model_path = Path(args.model_path) if args.model_path else default_autospeed_checkpoint(zod)
    img_dir = get_images_blur_dir(zod, seq)
    calib_path = get_calibration_path(zod, seq)
    radar_dir = zod / "radar_front" / "sequences" / seq / "radar_front"

    if not img_dir.exists():
        print(f"Image dir not found: {img_dir}")
        return
    if not calib_path.exists():
        print(f"Calibration not found: {calib_path}")
        return

    # Associations
    assoc_path = zod / "associations" / f"{seq}_associations.json"
    if not assoc_path.exists():
        print(f"Run step1 first: python step1_timestamp_association.py --sequence {seq} --zod-root {zod}")
        return

    with open(assoc_path) as f:
        assoc = json.load(f)
    with open(calib_path) as f:
        calib = json.load(f)["FC"]
    W, H = calib["image_dimensions"][0], calib["image_dimensions"][1]
    hfov_deg = calib["field_of_view"][0]
    cam_ext = np.array(calib["extrinsics"])
    radar_ext = np.array(calib["radar_extrinsics"])

    radar_data = np.load(assoc["radar_npy_path"], allow_pickle=True)
    model = AutoSpeedNetworkInfer(str(model_path))

    out_path = Path(args.output) if args.output else sequence_output_dir(zod, seq) / "cipo_radar.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_saved = False

    # Pass 1: forward pass - cluster matches only (no forward tracking)
    n_assoc = len(assoc["associations"])
    results = []
    for idx, rec in enumerate(assoc["associations"]):
        img_path = img_dir / rec["image"]
        ts_ns = rec.get("image_timestamp_ns")
        if not img_path.exists():
            results.append({
                "image": rec["image"], "cipo_detected": False,
                "distance_m": None, "speed_ms": None,
                "pixel_point": None, "bev_xy": None, "speed_ms_adjusted": None,
                **_rec_meta(rec)
            })
            continue

        img = Image.open(img_path).convert("RGB")
        # Center-crop to 50° horizontal FOV then run AutoSpeedNetworkInfer (letterboxes to 1024×512 internally)
        crop_img, crop_info = center_crop_50deg_resize(img, W, H, hfov_deg)
        if not sample_saved:
            sample_saved = True
            sample_path = str(out_path.parent / "model_input_sample.png")
            crop_img.save(sample_path)  # already 1024×512 — direct save
            print(f"Saved model input sample (50°-crop direct-resized 1024x512) -> {sample_path}")
        preds = model.inference(crop_img)
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  CIPO-radar: {idx + 1}/{n_assoc} images", flush=True)

        # CIPO: Level 1 and 2 only (dangerous/most in path). Exclude Level 3 (cyan, less in-path).
        CIPO_CLASSES = (1, 2)
        cipo = [p for p in preds if int(p[5]) in CIPO_CLASSES]
        if not cipo:
            curvature = rec.get("curvature_inv_m", 0.0)
            cluster = None
            az_result_deg = None
            cipo_scenario = None

            # Scenario 2: other bboxes present - find first bbox whose radar cluster is on path
            if preds:
                clusters_s2 = get_radar_clusters(radar_data, rec["radar_timestamp_ns"], lat_buffer=_LAT_BUFFER_M)
                cluster_s2, az_s2 = find_cipo_via_bbox(
                    preds, curvature, clusters_s2, cam_ext, radar_ext, crop_info,
                    lat_buffer_m=_LAT_BUFFER_M, path_buffer_m=_LAT_BUFFER_PATH_M,
                )
                if cluster_s2 is not None:
                    cluster = cluster_s2
                    az_result_deg = float(np.rad2deg(az_s2))
                    cipo_scenario = 2

            # Scenario 3: no bbox overlap - path search, require moving object
            # Radar range_rate is relative; object_speed_world = range_rate + ego_speed. Static: |...| < threshold.
            if cluster is None:
                ego_speed = rec.get("ego_speed_ms")
                cluster_s3 = find_cluster_on_path_direct(
                    radar_data, rec["radar_timestamp_ns"], curvature,
                    lat_buffer_m=_LAT_BUFFER_PATH_M,
                    min_abs_speed_world=_MIN_ABS_SPEED_WORLD_MS,
                    ego_speed_ms=ego_speed,
                    min_abs_range_rate=_MIN_ABS_RANGE_RATE_FALLBACK if ego_speed is None else None,
                )
                if cluster_s3 is not None:
                    cluster = cluster_s3
                    az_result_deg = float(np.rad2deg(_path_azimuth_at_range(curvature, cluster["range"])))
                    cipo_scenario = 3

            if cluster is not None:
                viz = _viz_fields_from_cluster(cluster)
                results.append({
                    "image": rec["image"],
                    "cipo_detected": False,
                    "cipo_from_path": True,
                    "cipo_scenario": cipo_scenario,
                    "azimuth_radar_deg": az_result_deg,
                    "distance_m": round(cluster["range"], 2),
                    "speed_ms": round(cluster["range_rate"], 2),
                    "pixel_point": None,
                    "bev_xy": viz.get("bev_xy"),
                    "speed_ms_adjusted": viz.get("speed_ms_adjusted"),
                    **_rec_meta(rec),
                })
            else:
                results.append({
                    "image": rec["image"], "cipo_detected": False,
                    "distance_m": None, "speed_ms": None,
                    "pixel_point": None, "bev_xy": None, "speed_ms_adjusted": None,
                    **_rec_meta(rec)
                })
            continue

        cipo.sort(key=lambda p: (p[1] + p[3]) / 2, reverse=True)
        x1, y1, x2, y2, conf, cls = cipo[0]
        u = (x1 + x2) / 2  # pixel x in 50°-crop space

        # Angle from 50°-crop optical axis → radar azimuth
        h_angle_deg = pixel_to_h_angle_deg_50(u, crop_info)
        az_radar = cam_dir_to_radar_azimuth(h_angle_deg, cam_ext, radar_ext)
        az_radar_deg = float(np.rad2deg(az_radar))

        # Map crop-space bbox back to full-image coordinates for storage
        fx1, fy1, fx2, fy2 = bbox_crop_to_full(x1, y1, x2, y2, crop_info)

        # CIPO: tighter vel_scale so points with similar range_rate cluster together
        clusters = get_radar_clusters(radar_data, rec["radar_timestamp_ns"], lat_buffer=_LAT_BUFFER_M, vel_scale=1.0)
        cluster = find_nearest_cluster_lateral(clusters, az_radar, lat_buffer_m=_LAT_BUFFER_M)
        relaxed_cone_used = False
        if cluster is None:
            cluster = find_nearest_cluster_lateral(clusters, az_radar, lat_buffer_m=_LAT_BUFFER_RELAXED_M)
            relaxed_cone_used = cluster is not None
        cipo_from_path = False
        track_from_prev = False

        if cluster is None:
            # 1. Track from previous frames only (same pass; future frames not in `results` yet).
            AZ_TOL_DEG = 4.0
            MAX_GAP_S = 1.0
            best_D, best_V, best_gap = None, None, float("inf")
            for k in range(1, min(TEMPORAL_NEIGHBOR_FRAMES + 1, len(results) + 1)):
                rj = results[-k]
                if rj.get("distance_m") is None:
                    continue
                ts_j = rj.get("image_timestamp_ns")
                az_j = rj.get("azimuth_radar_deg")
                if ts_j is None or az_j is None:
                    continue
                dt_s = (ts_ns - ts_j) / 1e9
                if dt_s <= 0 or dt_s > MAX_GAP_S:
                    continue
                daz = abs(np.angle(np.exp(1j * np.deg2rad(az_radar_deg - az_j))))
                if np.rad2deg(daz) > AZ_TOL_DEG:
                    continue
                D_est = rj["distance_m"] + rj["speed_ms"] * dt_s
                if D_est <= 0:
                    continue
                if k < best_gap:
                    best_gap = k
                    best_D = D_est
                    best_V = rj["speed_ms"]
            if best_D is not None:
                cluster = {"range": best_D, "range_rate": best_V}
                track_from_prev = True

        if cluster is None:
            # 2. Fallback: path-based clustering (curvature)
            curvature = rec.get("curvature_inv_m", 0.0)
            cluster = find_nearest_cluster_on_path(clusters, curvature, lat_buffer_m=_LAT_BUFFER_M)
            if cluster is not None:
                cipo_from_path = True
                az_path_rad = _path_azimuth_at_range(curvature, cluster["range"])
                az_radar_deg = float(np.rad2deg(az_path_rad))

        if cluster is not None:
            D, V = cluster["range"], cluster["range_rate"]
            viz = _viz_fields_from_cluster(cluster, azimuth_rad_deg=az_radar_deg if cluster.get("azimuth") is None else None)
            bbox = [fx1, fy1, fx2, fy2]
            out = {
                "image": rec["image"],
                "cipo_detected": True,
                "bbox": bbox,
                "azimuth_radar_deg": az_radar_deg,
                "distance_m": round(D, 2),
                "speed_ms": round(V, 2),
                "pixel_point": _pixel_point_from_bbox(bbox),
                "bev_xy": viz.get("bev_xy"),
                "speed_ms_adjusted": viz.get("speed_ms_adjusted"),
                **_rec_meta(rec),
            }
            if cipo_from_path:
                out["cipo_from_path"] = True
            if track_from_prev:
                out["track_from_prev"] = True
            if relaxed_cone_used:
                out["relaxed_cone_used"] = True
            results.append(out)
        else:
            bbox = [fx1, fy1, fx2, fy2]
            results.append({
                "image": rec["image"],
                "cipo_detected": True,
                "bbox": bbox,
                "azimuth_radar_deg": az_radar_deg,
                "distance_m": None,
                "speed_ms": None,
                "pixel_point": _pixel_point_from_bbox(bbox),
                "bev_xy": None,
                "speed_ms_adjusted": None,
                **_rec_meta(rec),
            })

    print(f"  Pass 1 done: {len(results)} frames. Running backfill...", flush=True)
    # Pass 2: CIPO frames with bbox but no radar match — look ahead AND behind TEMPORAL_NEIGHBOR_FRAMES
    LOOKAHEAD = TEMPORAL_NEIGHBOR_FRAMES
    LOOKBEHIND = TEMPORAL_NEIGHBOR_FRAMES
    AZ_TOL_DEG = 4.0  # same object: azimuth within 4 deg
    MAX_GAP_S = 1.0   # max time gap (seconds)

    for i in range(len(results)):
        r = results[i]
        if not r.get("cipo_detected") or r.get("distance_m") is not None:
            continue
        ts_i = r.get("image_timestamp_ns")
        az_i = r.get("azimuth_radar_deg")
        if ts_i is None or az_i is None:
            continue

        best_D, best_V, best_gap = None, None, float("inf")

        def check_match(rj, ts_j, az_j, dt_s, D_ref, V_ref, is_forward):
            """is_forward: ref is from past -> D_est = D_ref + V_ref*dt"""
            if ts_j is None or D_ref is None or V_ref is None:
                return None, None
            if dt_s <= 0 or dt_s > MAX_GAP_S:
                return None, None
            daz = abs(np.angle(np.exp(1j * np.deg2rad(az_i - az_j))))
            if np.rad2deg(daz) > AZ_TOL_DEG:
                return None, None
            if is_forward:
                D_est = D_ref + V_ref * dt_s
            else:
                D_est = D_ref - V_ref * dt_s
            if D_est <= 0:
                return None, None
            return D_est, V_ref

        # Look ahead (future frames): D(t) = D_future - V*dt
        for k in range(1, LOOKAHEAD + 1):
            j = i + k
            if j >= len(results):
                break
            rj = results[j]
            if rj.get("distance_m") is None:
                continue
            ts_j = rj.get("image_timestamp_ns")
            az_j = rj.get("azimuth_radar_deg")
            dt_s = (ts_j - ts_i) / 1e9
            D_est, V_est = check_match(rj, ts_j, az_j, dt_s, rj["distance_m"], rj["speed_ms"], is_forward=False)
            if D_est is not None and k < best_gap:
                best_gap = k
                best_D = D_est
                best_V = V_est

        # Look behind (past frames): D(t) = D_past + V*dt
        for k in range(1, LOOKBEHIND + 1):
            j = i - k
            if j < 0:
                break
            rj = results[j]
            if rj.get("distance_m") is None:
                continue
            ts_j = rj.get("image_timestamp_ns")
            az_j = rj.get("azimuth_radar_deg")
            dt_s = (ts_i - ts_j) / 1e9
            D_est, V_est = check_match(rj, ts_j, az_j, dt_s, rj["distance_m"], rj["speed_ms"], is_forward=True)
            if D_est is not None and k < best_gap:
                best_gap = k
                best_D = D_est
                best_V = V_est

        if best_D is not None:
            r["distance_m"] = round(best_D, 2)
            r["speed_ms"] = round(best_V, 2)
            viz = _viz_fields_from_cluster({"range": best_D, "range_rate": best_V}, azimuth_rad_deg=az_i)
            r["bev_xy"] = viz.get("bev_xy")
            r["speed_ms_adjusted"] = viz.get("speed_ms_adjusted")

    # Pass 3: no-CIPO — path/radar association via neighbors ±TEMPORAL_NEIGHBOR_FRAMES (iterative)
    NO_CIPO_LOOKAHEAD = TEMPORAL_NEIGHBOR_FRAMES
    NO_CIPO_LOOKBEHIND = TEMPORAL_NEIGHBOR_FRAMES
    NO_CIPO_MAX_GAP_S = 1.0
    NO_CIPO_RANGE_TOL_M = 3.0
    NO_CIPO_VEL_TOL_MS = 2.0
    NO_CIPO_MAX_ITER = 5

    n_filled = 1
    iter_count = 0
    while n_filled > 0 and iter_count < NO_CIPO_MAX_ITER:
        n_filled = 0
        iter_count += 1
        for i in range(len(results)):
            r = results[i]
            if r.get("cipo_detected") or r.get("distance_m") is not None:
                continue
            ts_i = r.get("image_timestamp_ns")
            if ts_i is None:
                continue
            rec = assoc["associations"][i]
            curvature = rec.get("curvature_inv_m", 0.0)

            best_cluster, best_gap = None, float("inf")

            def try_neighbor(rj, ts_j, dt_s, D_ref, V_ref, is_forward):
                if ts_j is None or D_ref is None or V_ref is None:
                    return None
                if dt_s <= 0 or dt_s > NO_CIPO_MAX_GAP_S:
                    return None
                D_est = D_ref + V_ref * dt_s if is_forward else D_ref - V_ref * dt_s
                if D_est <= 0:
                    return None
                return find_cluster_on_path_direct(
                    radar_data, rec["radar_timestamp_ns"], curvature,
                    lat_buffer_m=_LAT_BUFFER_PATH_M,
                    D_est=D_est, range_tol_m=NO_CIPO_RANGE_TOL_M,
                    V_ref=V_ref, vel_tol_ms=NO_CIPO_VEL_TOL_MS,
                    min_pts=1,  # 1 point OK: neighbor frames already confirmed the object
                )

            for k in range(1, NO_CIPO_LOOKAHEAD + 1):
                j = i + k
                if j >= len(results):
                    break
                rj = results[j]
                if rj.get("distance_m") is None:
                    continue
                ts_j = rj.get("image_timestamp_ns")
                dt_s = (ts_j - ts_i) / 1e9
                cluster = try_neighbor(rj, ts_j, dt_s, rj["distance_m"], rj["speed_ms"], is_forward=False)
                if cluster is not None and k < best_gap:
                    best_gap = k
                    best_cluster = cluster

            for k in range(1, NO_CIPO_LOOKBEHIND + 1):
                j = i - k
                if j < 0:
                    break
                rj = results[j]
                if rj.get("distance_m") is None:
                    continue
                ts_j = rj.get("image_timestamp_ns")
                dt_s = (ts_i - ts_j) / 1e9
                cluster = try_neighbor(rj, ts_j, dt_s, rj["distance_m"], rj["speed_ms"], is_forward=True)
                if cluster is not None and k < best_gap:
                    best_gap = k
                    best_cluster = cluster

            if best_cluster is not None:
                az_path_rad = _path_azimuth_at_range(curvature, best_cluster["range"])
                r["distance_m"] = round(best_cluster["range"], 2)
                r["speed_ms"] = round(best_cluster["range_rate"], 2)
                r["azimuth_radar_deg"] = float(np.rad2deg(az_path_rad))
                viz = _viz_fields_from_cluster(best_cluster)
                r["bev_xy"] = viz.get("bev_xy")
                r["speed_ms_adjusted"] = viz.get("speed_ms_adjusted")
                r["cipo_from_path"] = True
                r["track_from_neighbor"] = True
                n_filled += 1

        if n_filled > 0:
            print(f"  No-CIPO temporal fill iter {iter_count}: filled {n_filled} frames", flush=True)

    with open(out_path, "w") as f:
        json.dump({"sequence": seq, "results": results}, f, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
