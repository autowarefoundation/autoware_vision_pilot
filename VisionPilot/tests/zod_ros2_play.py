#!/usr/bin/env python3
"""Play one official ZOD sequence into VisionPilot over ROS2.

VisionPilot stays on the stock 2MP path (1920×1080 → 2:1 top-crop → 1024×512).
This script is the 8MP adapter: center-crop front_blur to 50° HFOV (same
formula as Models/data_utils/load_data_auto_drive.py), resize to 1920×1080,
then attach the nearest radar scan and ego speed to each camera frame.

Give someone this file and a ZOD download. They only pass --zod-root and --seq.

Required ZOD layout (official modality-split download, not a per-seq bundle):

    {zod-root}/
      infos/sequences/{SEQ}/info.json
      infos/sequences/{SEQ}/ego_motion.json          # speed = ||velocity||
      infos/sequences/{SEQ}/calibration.json         # optional; HFOV fallback 120°
      images_blur_000000_000490/sequences/{SEQ}/camera_front_blur/*.jpg
      images_blur_000491_000981/sequences/{SEQ}/camera_front_blur/*.jpg
      images_blur_000982_001472/sequences/{SEQ}/camera_front_blur/*.jpg
      radar_front/sequences/{SEQ}/radar_front/*.npy  # all scans in one file

  SEQ is six digits, e.g. 000000. Images live in the shard that covers that id.

Topics (VisionPilot defaults):
  /camera/image          sensor_msgs/Image          bgr8 1920×1080
  /radar/points          sensor_msgs/PointCloud2    range, azimuth, range_rate
  /vehicle/speed         std_msgs/Float64           m/s, best-effort

Speed comes from ego_motion.json (same clock as the camera). vehicle_data.hdf5
and oxts.hdf5 are not used.

Example
  source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=17
  python3 tests/zod_ros2_play.py --zod-root /path/to/zod --seq 000000 --rate 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64, Header

# Stock VisionPilot camera. 50° crop then this size → VP 2:1 top-crop.
OUT_W, OUT_H = 1920, 1080
TARGET_HFOV_DEG = 50.0
# Matches Models/data_utils/load_data_auto_drive.py _ZOD_HFOV_DEG. Lock this
# (not per-seq calib) so the published 2MP window matches config/H.yaml.
DEFAULT_CAM_HFOV_DEG = 120.0

# Full-frame ZOD H (3848×2168 → ground). Composed with the 8MP→2MP crop for --write-h.
H_8MP = np.array(
    [
        [1.3383859768509865e-03, -6.0194928664714100e-04, -1.0339601516723633e01],
        [4.2337169870734215e-03, -2.2559992794413120e-04, -7.9301109313964844e00],
        [7.6705691753886640e-05, -2.6293119881302120e-03, 2.7097563743591310e00],
    ],
    dtype=np.float64,
)

IMAGES_BLUR_RANGES = (
    (0, 490, "images_blur_000000_000490"),
    (491, 981, "images_blur_000491_000981"),
    (982, 1472, "images_blur_000982_001472"),
)


def iso_to_ns(stamp: str) -> int:
    stamp = stamp.replace("Z", "+00:00")
    dt = datetime.fromisoformat(stamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def ns_to_time(ns: int) -> Time:
    t = Time()
    t.sec = int(ns // 1_000_000_000)
    t.nanosec = int(ns % 1_000_000_000)
    return t


def nearest_index(sorted_ns: np.ndarray, stamp_ns: int) -> int:
    i = int(np.searchsorted(sorted_ns, stamp_ns))
    if i <= 0:
        return 0
    if i >= len(sorted_ns):
        return len(sorted_ns) - 1
    return i if abs(int(sorted_ns[i]) - stamp_ns) < abs(int(sorted_ns[i - 1]) - stamp_ns) else i - 1


def images_blur_dir(zod_root: Path, seq: str) -> Path:
    seq_int = int(seq)
    for lo, hi, folder in IMAGES_BLUR_RANGES:
        if lo <= seq_int <= hi:
            return zod_root / folder / "sequences" / seq / "camera_front_blur"
    return zod_root / "images_blur_000000_000490" / "sequences" / seq / "camera_front_blur"


def resolve_image(zod_root: Path, seq: str, filepath: str) -> Path:
    """info.json filepath is sequences/{seq}/camera_front_blur/{name}.jpg."""
    rel = Path(filepath)
    name = rel.name
    direct = images_blur_dir(zod_root, seq) / name
    if direct.is_file():
        return direct
    matches = list(zod_root.glob(f"images*/{filepath}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"camera frame not found: {filepath}")


def model_input_crop(width: int, height: int, camera_hfov_deg: float) -> tuple[int, int, int, int]:
    """50° center crop at 16:9 so the resize to 1920×1080 is undistorted.

    Horizontal span matches load_data_auto_drive (width * 50 / hfov).
    Height is 16:9 of that width (native 2MP), not the training 2:1 band.
    VisionPilot then top-crops 1920×1080 → 1920×960 → 1024×512.
    """
    cam = max(float(camera_hfov_deg), 1.0)
    crop_w = int(round(width * TARGET_HFOV_DEG / cam))
    crop_w = min(max(crop_w, 2), width)
    crop_h = int(round(crop_w * OUT_H / OUT_W))
    crop_h = min(max(crop_h, 2), height)
    crop_x = max(0, (width - crop_w) // 2)
    crop_y = max(0, (height - crop_h) // 2)
    crop_w = min(crop_w, width - crop_x)
    crop_h = min(crop_h, height - crop_y)
    return crop_x, crop_y, crop_w, crop_h


def to_2mp(bgr: np.ndarray, camera_hfov_deg: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    x, y, cw, ch = model_input_crop(w, h, camera_hfov_deg)
    return cv2.resize(bgr[y : y + ch, x : x + cw], (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)


def compose_h_2mp(raw_w: int, raw_h: int, camera_hfov_deg: float) -> np.ndarray:
    x, y, cw, ch = model_input_crop(raw_w, raw_h, camera_hfov_deg)
    t = np.array([[cw / OUT_W, 0.0, x], [0.0, ch / OUT_H, y], [0.0, 0.0, 1.0]], dtype=np.float64)
    return H_8MP @ t


def write_h_yaml(path: Path, h: np.ndarray, raw_w: int, raw_h: int, camera_hfov_deg: float) -> None:
    x, y, cw, ch = model_input_crop(raw_w, raw_h, camera_hfov_deg)
    r0 = ", ".join(f"{v:.16e}" for v in h[0])
    r1 = ", ".join(f"{v:.16e}" for v in h[1])
    r2 = ", ".join(f"{v:.16e}" for v in h[2])
    path.write_text(
        "%YAML:1.0\n"
        "---\n"
        f"# ZOD front_blur {raw_w}×{raw_h} @ {camera_hfov_deg:.2f}° → 50° crop "
        f"{cw}×{ch} @ ({x},{y}) → {OUT_W}×{OUT_H}.\n"
        "# H_2mp = H_8mp × T. Not the OpenLane 1920×1080 matrix.\n"
        "# MODIFY THIS MATRIX FOR DIFFERENT INPUT CAMERA!!!\n"
        "H: !!opencv-matrix\n"
        "  rows: 3\n"
        "  cols: 3\n"
        "  dt: d\n"
        f"  data: [ {r0},\n"
        f"          {r1},\n"
        f"          {r2} ]\n"
    )


def load_sequence(zod_root: Path, seq: str, camera_hfov_deg: float):
    seq = seq.zfill(6)
    info_path = zod_root / "infos" / "sequences" / seq / "info.json"
    ego_path = zod_root / "infos" / "sequences" / seq / "ego_motion.json"
    radar_dir = zod_root / "radar_front" / "sequences" / seq / "radar_front"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}")
    if not ego_path.is_file():
        raise FileNotFoundError(f"missing {ego_path} (speed source)")
    radar_files = sorted(radar_dir.glob("*.npy"))
    if not radar_files:
        raise FileNotFoundError(f"missing radar npy under {radar_dir}")

    info = json.loads(info_path.read_text())
    frames = info["camera_frames"]["front_blur"]
    cam = []
    for fr in frames:
        path = resolve_image(zod_root, seq, fr["filepath"])
        cam.append((iso_to_ns(fr["time"]), path))

    radar = np.load(radar_files[0])
    scan_ts = np.unique(radar["timestamp"])
    scans = {int(t): radar[radar["timestamp"] == t] for t in scan_ts}

    ego = json.loads(ego_path.read_text())
    ego_ts = np.rint(np.asarray(ego["timestamps"], dtype=np.float64) * 1e9).astype(np.int64)
    ego_speed = np.linalg.norm(np.asarray(ego["velocities"], dtype=np.float64), axis=1)

    return cam, scan_ts.astype(np.int64), scans, ego_ts, ego_speed


def radar_cloud(scan: np.ndarray, stamp_ns: int, frame_id: str = "radar") -> PointCloud2:
    pts = np.column_stack(
        (
            scan["radar_range"].astype(np.float32),
            scan["azimuth_angle"].astype(np.float32),
            scan["range_rate"].astype(np.float32),
        )
    )
    header = Header()
    header.stamp = ns_to_time(stamp_ns)
    header.frame_id = frame_id
    fields = [
        PointField(name="range", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="azimuth", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="range_rate", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    return point_cloud2.create_cloud(header, fields, pts.tolist())


class ZodPlayer(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("zod_ros2_play")
        self.args = args
        self.bridge = CvBridge()

        cam_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        radar_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        speed_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_cam = self.create_publisher(Image, args.camera_topic, cam_qos)
        self.pub_radar = self.create_publisher(PointCloud2, args.radar_topic, radar_qos)
        self.pub_speed = self.create_publisher(Float64, args.speed_topic, speed_qos)

        self.cam, self.scan_ts, self.scans, self.ego_ts, self.ego_speed = load_sequence(
            args.zod_root, args.seq, args.hfov_deg
        )
        if not self.cam:
            raise RuntimeError("no camera frames")

        first = cv2.imread(str(self.cam[0][1]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"failed to read {self.cam[0][1]}")
        raw_h, raw_w = first.shape[:2]
        crop = model_input_crop(raw_w, raw_h, args.hfov_deg)
        self.get_logger().info(
            f"seq={args.seq.zfill(6)}  frames={len(self.cam)}  "
            f"radar_scans={len(self.scan_ts)}  ego={len(self.ego_ts)}  "
            f"raw={raw_w}x{raw_h}  crop={crop[2]}x{crop[3]}@{crop[0]},{crop[1]}  "
            f"pub={OUT_W}x{OUT_H}  hfov={args.hfov_deg:.2f}→{TARGET_HFOV_DEG:.0f}"
        )

        if args.write_h:
            h = compose_h_2mp(raw_w, raw_h, args.hfov_deg)
            write_h_yaml(args.write_h, h, raw_w, raw_h, args.hfov_deg)
            self.get_logger().info(f"wrote H.yaml {args.write_h} — rebuild VisionPilot to refresh C")

    def play(self) -> None:
        n = len(self.cam)
        t0_wall = time.monotonic()
        t0_ns = self.cam[0][0]
        rate = max(self.args.rate, 1e-6)

        for i, (cam_ns, path) in enumerate(self.cam):
            if not rclpy.ok():
                break
            due = t0_wall + (cam_ns - t0_ns) / 1e9 / rate
            sleep = due - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().warn(f"skip unreadable {path}")
                continue
            frame = to_2mp(bgr, self.args.hfov_deg)

            ri = nearest_index(self.scan_ts, cam_ns)
            r_ns = int(self.scan_ts[ri])
            scan = self.scans[r_ns]
            si = nearest_index(self.ego_ts, cam_ns)
            speed = float(self.ego_speed[si])

            # Lock all three to the camera stamp so VP take_closest always hits.
            speed_msg = Float64()
            speed_msg.data = speed
            self.pub_speed.publish(speed_msg)
            self.pub_radar.publish(radar_cloud(scan, cam_ns))

            img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img.header.stamp = ns_to_time(cam_ns)
            img.header.frame_id = "camera"
            self.pub_cam.publish(img)

            if i == 0 or (i + 1) % 20 == 0 or i + 1 == n:
                dt_ms = (r_ns - cam_ns) / 1e6
                self.get_logger().info(
                    f"[{i+1}/{n}] {path.name}  "
                    f"{frame.shape[1]}x{frame.shape[0]}  "
                    f"radar={len(scan)} pts Δ={dt_ms:+.1f}ms  "
                    f"ego_speed={speed:.2f} m/s"
                )
            rclpy.spin_once(self, timeout_sec=0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zod-root", type=Path, default=Path("/home/pranavdoma/Downloads/data/zod"))
    p.add_argument("--seq", default="000000", help="sequence id, e.g. 000000 or 001411")
    p.add_argument("--rate", type=float, default=1.0, help="playback rate (1 = realtime)")
    p.add_argument(
        "--hfov-deg",
        type=float,
        default=DEFAULT_CAM_HFOV_DEG,
        help="raw camera HFOV used for the 50° crop; keep 120 to match H.yaml",
    )
    p.add_argument("--camera-topic", default="/camera/image")
    p.add_argument("--radar-topic", default="/radar/points")
    p.add_argument("--speed-topic", default="/vehicle/speed")
    p.add_argument(
        "--write-h",
        type=Path,
        default=None,
        help="write the matching 2MP H.yaml (then rebuild VisionPilot)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.zod_root = args.zod_root.expanduser().resolve()
    args.seq = args.seq.zfill(6)
    rclpy.init()
    node = ZodPlayer(args)
    try:
        node.play()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
