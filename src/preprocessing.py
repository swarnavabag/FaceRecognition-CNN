import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from src.shared.config import (
    TRAIN_RAW_DIR,
    VAL_RAW_DIR,
    TRAIN_PROCESSED_DIR,
    VAL_PROCESSED_DIR,
    IMAGE_SIZE
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
MARGIN = 0.2
JPEG_QUALITY = 95


# ============================================================
# DETECTOR
# ============================================================

detector = MTCNN(
    keep_all=True,
    device=DEVICE
)


# ============================================================
# FACE UTILITIES
# ============================================================

def get_main_face_box(boxes: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Select the largest detected face.
    """
    if boxes is None:
        return None

    if len(boxes) == 1:
        return boxes[0]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[np.argmax(areas)]


def crop_face(img: np.ndarray, bbox: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """
    Crop face with additional margin around bounding box.
    """
    h, w = img.shape[:2]

    x1, y1, x2, y2 = bbox.astype(int)

    box_w = x2 - x1
    box_h = y2 - y1

    dx = int(box_w * margin)
    dy = int(box_h * margin)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)

    return img[y1:y2, x1:x2]


def get_border_mean_color(img: np.ndarray, border: int = 5):
    """
    Compute average border color for natural-looking padding.
    """
    h, w = img.shape[:2]

    top = img[:border, :]
    bottom = img[h - border:, :]
    left = img[:, :border]
    right = img[:, w - border:]

    border_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )

    return border_pixels.mean(axis=0).astype(np.uint8).tolist()


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """
    Pad image to square shape.
    """
    h, w = img.shape[:2]

    if h == w:
        return img

    size = max(h, w)

    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top

    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    pad_color = get_border_mean_color(img)

    padded = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    return padded


# ============================================================
# PREPROCESS PIPELINE
# ============================================================

def preprocess_image(
    img_path: Path,
    detector: MTCNN,
    margin: float = MARGIN,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for one image.
    """
    img = cv2.imread(str(img_path))

    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, _ = detector.detect(img_rgb)

    if boxes is None:
        return None

    bbox = get_main_face_box(boxes)
    cropped = crop_face(img, bbox, margin)
    padded = pad_to_square(cropped)
    resized = cv2.resize(padded, target_size)

    return resized


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(
    src_root: Path,
    dst_root: Path,
    detector: MTCNN,
    max_folders=None,
):
    """
    Process an entire dataset directory.
    """
    total = 0
    processed = 0
    skipped = 0
    failed_files = []

    start_time = time.time()

    identity_folders = [f for f in src_root.iterdir() if f.is_dir()]

    if max_folders is not None:
        identity_folders = identity_folders[:max_folders]

    total_folders = len(identity_folders)

    for idx, folder in enumerate(identity_folders):
        dst_folder = dst_root / folder.name

        # Resume support
        if dst_folder.exists() and any(dst_folder.iterdir()):
            print(f"[{idx+1}/{total_folders}] Skipping existing folder: {folder.name}")
            continue

        dst_folder.mkdir(parents=True, exist_ok=True)
        image_paths = list(folder.iterdir())

        for image_path in image_paths:
            total += 1

            processed_img = preprocess_image(image_path, detector)

            if processed_img is None:
                skipped += 1
                failed_files.append(str(image_path))
                continue

            save_path = dst_folder / (image_path.stem + ".jpg")

            cv2.imwrite(
                str(save_path),
                processed_img,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            processed += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == total_folders:
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0

            print(
                f"[{idx+1}/{total_folders}] "
                f"Processed: {processed} | "
                f"Skipped: {skipped} | "
                f"Speed: {speed:.2f} img/s"
            )

    elapsed = time.time() - start_time

    print("\n========== DONE ==========")
    print(f"Elapsed time: {elapsed/60:.2f} minutes")
    print(f"Total images seen: {total}")
    print(f"Processed images: {processed}")
    print(f"Skipped images: {skipped}")

    if failed_files:
        failed_path = dst_root / "failed_files.txt"

        with open(failed_path, "w", encoding="utf-8") as f:
            for file in failed_files:
                f.write(file + "\n")

        print(f"Failed files saved to: {failed_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    TRAIN_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    process_dataset(TRAIN_RAW_DIR, TRAIN_PROCESSED_DIR, detector)
    process_dataset(VAL_RAW_DIR, VAL_PROCESSED_DIR, detector)


if __name__ == "__main__":
    main()