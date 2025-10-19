# Original code
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/align/face_align.py

import argparse
import os

import numpy as np
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
    IMAGE_PATH = "./data/actors/jeff"
    image_paths = [os.path.join(IMAGE_PATH, path) for path in os.listdir(IMAGE_PATH)]

    for image_path in tqdm(image_paths, desc="Cropping faces"):
        cropper(image_path)