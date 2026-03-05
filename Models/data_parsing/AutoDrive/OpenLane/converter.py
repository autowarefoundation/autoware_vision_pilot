import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm  # pip install tqdm
import json
import random
import numpy as np
from PIL import Image

orig_image_width = 1920
orig_image_height = 1280
new_image_width = 1024
new_image_height = 512


def move_images(input_dir, output_dir):
    input_dir = Path(input_dir)

    files = [f for f in input_dir.rglob("*") if f.is_file()]

    for file in tqdm(files, desc="Moving files", unit="file"):
        if file.is_file():
            target = Path(output_dir) / file.name
            target.parent.mkdir(parents=True, exist_ok=True)

            # If file with same name exists, rename
            if target.exists():
                print(f"File with same name already exists: {target}")

            # Move file
            shutil.move(str(file), str(target))

    if input_dir.exists() and input_dir.is_dir():
        shutil.rmtree(input_dir)


def process_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = [f for f in input_dir.rglob("*") if f.is_file()]

    for file in tqdm(files, desc="Cropping files", unit="file"):
        try:
            with Image.open(file) as img:
                width, height = img.size
                # Crop: (left, upper, right, lower)
                cropped = img.crop((0, 320, width, height))
                # Resize
                resized = cropped.resize((1024, 512), Image.LANCZOS)

                # Save to output directory
                target = output_dir / file.name
                target.parent.mkdir(parents=True, exist_ok=True)

                # If file with same name exists, rename
                if target.exists():
                    print(f"File with same name already exists: {target}")
                    # optional: add suffix
                    stem, ext = file.stem, file.suffix
                    target = output_dir / f"{stem}_cropped{ext}"

                resized.save(target)

        except Exception as e:
            print(f"Failed to process {file}: {e}")

    # Optionally, delete the original input directory
    if input_dir.exists() and input_dir.is_dir():
        shutil.rmtree(input_dir)


def convert_labels(input_dir, output_dir):
    crop_top = 320

    input_dir = Path(input_dir)
    files = [f for f in input_dir.rglob("*") if f.is_file()]

    new_height = orig_image_height - crop_top

    for file in tqdm(files, desc="Convert labels", unit="file"):
        base_name = file.name.split(".", 1)[0]
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            labels = []

            for box in data["result"]:
                id = box.get("id", box.get("attribute"))
                id = 3 if str(id) == "4" else id

                # Adjust y because top was cropped
                y_top = float(box["y"]) - crop_top

                # Skip boxes fully removed by crop
                if y_top + float(box["height"]) <= 0:
                    continue

                # Clamp to image
                y_top = max(0, y_top)

                # Normalize
                width = float(box["width"]) / orig_image_width
                height = float(box["height"]) / new_height
                x = (float(box["x"]) + float(box["width"]) / 2) / orig_image_width
                y = (y_top + float(box["height"]) / 2) / new_height

                labels.append([id, x, y, width, height])

        target = Path(output_dir) / f"{base_name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", encoding="utf-8") as f:
            for item in labels:
                f.write(" ".join(map(str, item)) + "\n")

        file.unlink()

    shutil.rmtree(input_dir)


def convert_lane3d_labels(input_dir, output_dir):
    input_dir = Path(input_dir)
    files = [f for f in input_dir.rglob("*") if f.is_file()]

    for file in tqdm(files, desc="Convert 3dlane labels", unit="file"):
        base_name = file.name.split(".", 1)[0]
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

            for lane_line in data['lane_lines']:
                # Convert to Nx2 points
                pts = np.array(lane_line['uv']).T.astype(np.float32)  # shape: (N, 2)
                # Shift y coordinates (v axis)
                pts[:, 1] -= orig_image_height - (orig_image_width / 2)
                # Scale x and y
                pts *= new_image_width / orig_image_width
                # Convert back to original format (2, N)
                lane_line['uv'] = pts.T.tolist()

            target = Path(output_dir) / f"{base_name}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)

            with target.open("w", encoding="utf-8") as f:
                json.dump(data, f)

        file.unlink()

    shutil.rmtree(input_dir)


def expand_training_set(dataset_dir, fract=0.25):
    val_images_dir = dataset_dir + "/images/val"
    val_labels_dir = dataset_dir + "/labels/val"

    train_images_dir = dataset_dir + "/images/train"
    train_labels_dir = dataset_dir + "/labels/train"

    val_images = [f for f in Path(val_images_dir).rglob("*") if f.is_file()]
    random.shuffle(val_images)

    split_idx = int(len(val_images) * fract)
    # val_files = files[:split_idx]
    train_images = val_images[split_idx:]

    for image in tqdm(train_images, desc="Expand training dataset", unit="file"):
        if image.is_file():
            target_image = Path(train_images_dir) / image.name
            label = Path(val_labels_dir) / f"{image.stem}.txt"
            target_label = Path(train_labels_dir) / f"{image.stem}.txt"

            # Move file
            try:
                shutil.move(str(image), str(target_image))
                shutil.move(str(label), str(target_label))
            except Exception as e:
                print(f"Failed to move {image.name}: {e}")


def convert(dataset_dir):
    # convert training data
    input_training_dir = dataset_dir + "/images/training"
    output_training_dir = dataset_dir + "/images/train"
    process_images(input_training_dir, output_training_dir)

    input_dir = dataset_dir + "/labels/training"
    output_dir = dataset_dir + "/labels/train"
    convert_labels(input_dir, output_dir)

    # input_dir = dataset_dir + "/labels_lane3d/training"
    # output_dir = dataset_dir + "/labels_lane3d/train"
    # convert_lane3d_labels(input_dir, output_dir)

    # convert validation data
    input_dir = dataset_dir + "/images/validation"
    output_dir = dataset_dir + "/images/val"
    process_images(input_dir, output_dir)

    input_dir = dataset_dir + "/labels/validation"
    output_dir = dataset_dir + "/labels/val"
    convert_labels(input_dir, output_dir)

    # input_dir = dataset_dir + "/labels_lane3d/validation"
    # output_dir = dataset_dir + "/labels_lane3d/val"
    # convert_lane3d_labels(input_dir, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", help="dataset directory")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    convert(dataset_dir)
    # expand_training_set(dataset_dir)
