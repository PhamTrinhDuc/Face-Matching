# Simple face detection using MediaPipe
# Lightweight alternative to RetinaFace
# https://github.com/google/mediapipe

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import mediapipe as mp

FOLDER_SAVED = "./data/cropped/"

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# YuNet model path
YUNET_MODEL_PATH = "./models/face_detection_yunet_2023mar.onnx"
yunet_detector = None  # Lazy load

def ensure_yunet_model():
    """Download YuNet model if not exists"""
    import urllib.request
    
    os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
    
    if not os.path.exists(YUNET_MODEL_PATH):
        print("Downloading YuNet model (~300KB)...")
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            YUNET_MODEL_PATH
        )
        print("YuNet model ready!")


def get_yunet_detector(img_width, img_height):
    """Get YuNet detector instance"""
    global yunet_detector
    ensure_yunet_model()
    
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (img_width, img_height),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector


def cropper_medipipe(image_path: str):
    """Crop face from image using MediaPipe and return cropped face"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"{image_path} could not be read!")
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Detect faces
        results = face_detector.process(img_rgb)
        
        if not results.detections:
            print(f"No face detected in {image_path}")
            return None
        
        # Get the first (largest) face detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to pixel coordinates
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))
        
        # Add padding for better crop
        padding = int(min(x2 - x1, y2 - y1) * 0.1)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop face
        face_crop = img[y1:y2, x1:x2]
        
        # Save cropped image
        if not os.path.exists(FOLDER_SAVED):
            os.makedirs(FOLDER_SAVED)
        
        cropped_path = os.path.join(FOLDER_SAVED, os.path.basename(image_path))
        cv2.imwrite(cropped_path, face_crop)
        
        return face_crop
        
    except Exception as e:
        print(f"{image_path} is discarded due to exception: {e}")
        return None


def cropper_yunet(image_path: str):
    """Crop face using YuNet (OpenCV) - Fast & accurate"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"{image_path} could not be read!")
            return None
        
        h, w = img.shape[:2]
        
        # Get YuNet detector
        detector = get_yunet_detector(w, h)
        
        # Detect faces
        _, faces = detector.detect(img)
        
        if faces is None or len(faces) == 0:
            print(f"No face detected in {image_path}")
            return None
        
        # Get first face
        face = faces[0]
        
        # Extract bounding box (x, y, w, h)
        x, y, w_box, h_box = face[:4].astype(int)
        
        # Calculate x1, y1, x2, y2
        x1, y1 = x, y
        x2, y2 = x + w_box, y + h_box
        
        # Add minimal padding (10% instead of MediaPipe's large padding)
        padding = int(min(w_box, h_box) * 0.15)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop face
        face_crop = img[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            print(f"Crop failed for {image_path}")
            return None
        
        # Save cropped image
        if not os.path.exists(FOLDER_SAVED):
            os.makedirs(FOLDER_SAVED)
        
        cropped_path = os.path.join(FOLDER_SAVED, os.path.basename(image_path))
        cv2.imwrite(cropped_path, face_crop)
        
        return face_crop
        
    except Exception as e:
        print(f"{image_path} is discarded due to exception: {e}")
        return None


if __name__ == "__main__":
    IMAGE_PATH = "./data/student_images/B22DCCN050_743453_B198_81feeea2.jpg"
    
    # print("MediaPipe:")
    # cropper_medipipe(IMAGE_PATH)

    print("YuNet:")
    cropper_yunet(IMAGE_PATH)