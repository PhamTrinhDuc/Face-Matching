# Original code
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/align/face_align.py

import argparse
import os

import numpy as np
from align.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from PIL import Image
from tqdm import tqdm
from retinaface import RetinaFace

FOLDER_SAVED = "./cropped/"


def cropper(image_path: str):
    """Crop face from image and return 5 facial points"""
    try:  # Handle exception
        face_info = RetinaFace.detect_faces(img_path=image_path)
        print(face_info)

    except Exception:
        print(f"{image_path} is discarded due to exception!")
        return None
    x1, y1, x2, y2 = face_info['face_1']['facial_area']

    # Cắt ảnh
    img = Image.open(image_path)
    img = np.array(img)
    face_crop = img[y1:y2, x1:x2]
    Image.fromarray(face_crop).save(FOLDER_SAVED + os.path.basename(image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Alignment")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/",
        help="Directory with unaligned images.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.exists(data_root):
        raise ValueError(f"Data root {data_root} does not exist!")

    image_paths = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    print(f"Found {len(image_paths)} images in {data_root}")

    for image_path in tqdm(image_paths, desc="Cropping faces"):
        cropper(image_path)