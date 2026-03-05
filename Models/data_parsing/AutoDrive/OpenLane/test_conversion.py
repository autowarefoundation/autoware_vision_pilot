import cv2
import argparse
import json
import numpy as np
from pathlib import Path

LANE_COLOR_MAP = {
    0: (128, 128, 128),  # unknown → gray

    1: (255, 255, 255),  # white-dash
    2: (200, 200, 200),  # white-solid
    3: (255, 255, 255),  # double-white-dash
    4: (220, 220, 220),  # double-white-solid
    5: (180, 180, 180),  # white-ldash-rsolid
    6: (160, 160, 160),  # white-lsolid-rdash

    7: (0, 255, 255),  # yellow-dash
    8: (0, 200, 255),  # yellow-solid
    9: (0, 255, 255),  # double-yellow-dash
    10: (0, 220, 255),  # double-yellow-solid
    11: (0, 180, 255),  # yellow-ldash-rsolid
    12: (0, 160, 255),  # yellow-lsolid-rdash

    20: (255, 0, 0),  # left-curbside → blue
    21: (0, 0, 255),  # right-curbside → red
}


def draw_bbox(image_path):
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]

    label_path = Path(image_path.replace("images", "labels").replace(".jpg", ".txt"))
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines
            cls, x, y, w, h = map(float, parts)

            # Convert to pixel coordinates
            x_center_px = int(x * img_w)
            y_center_px = int(y * img_h)
            w_px = int(w * img_w)
            h_px = int(h * img_h)

            # Get top-left and bottom-right corners
            x1 = int(x_center_px - w_px / 2)
            y1 = int(y_center_px - h_px / 2)
            x2 = int(x_center_px + w_px / 2)
            y2 = int(y_center_px + h_px / 2)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    # Show image
    cv2.imshow("Image with bbox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def draw_lanes(image_path):
#     image = cv2.imread(image_path)
#     img_h, img_w = image.shape[:2]
#     pts = []
#
#     label_path = Path(image_path.replace("images", "labels_lane3d").replace(".jpg", ".txt"))
#     with label_path.open("r") as f:
#         data = json.load(f)
#
#         for lane_line in data['lane_lines']:
#             # Convert to Nx2 points
#             # pts = np.array(data['lane_lines'][0]['uv']).T.astype(np.int32)  # shape: (N, 2)
#             pts = np.array(lane_line['uv']).T.astype(np.float32)
#
#             # Shift y coordinates (v axis)
#             pts[:, 1] -= 320
#             # Scale x and y
#             pts *= 0.533333333
#             # Convert to int for OpenCV
#             pts = np.round(pts).astype(np.int32)
#
#
#             pts = pts[
#                 (pts[:, 0] >= 0) & (pts[:, 0] < img_w) &
#                 (pts[:, 1] >= 0) & (pts[:, 1] < img_h)
#                 ]
#
#             cv2.polylines(image, [pts], isClosed=False, color=LANE_COLOR_MAP[lane_line['category']], thickness=3)
#
#     # Show image
#     cv2.imshow("Image with bbox", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def draw_lanes(image_path):
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    pts = []

    label_path = Path(image_path.replace("images", "labels_lane3d").replace(".jpg", ".txt"))
    with label_path.open("r") as f:
        data = json.load(f)

        for lane_line in data['lane_lines']:
            # Convert to Nx2 points
            pts = np.array(lane_line['uv']).T.astype(np.float32)  # shape: (N, 2)
            pts = np.round(pts).astype(np.int32)

            pts = pts[
                (pts[:, 0] >= 0) & (pts[:, 0] < img_w) &
                (pts[:, 1] >= 0) & (pts[:, 1] < img_h)]

            cv2.polylines(image, [pts], isClosed=False, color=LANE_COLOR_MAP[lane_line['category']], thickness=3)

    # Show image
    cv2.imshow("Image with bbox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", help="image path")
    args = parser.parse_args()

    image = args.image_path
    # draw_bbox(image)
    draw_lanes(image)
