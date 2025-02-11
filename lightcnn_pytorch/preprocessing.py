import cv2
import dlib
import numpy as np
import torch
from typing import Union, Optional, Tuple
import os
from PIL import Image
from torchvision import transforms
from .utils import download_weights

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Will store predictor instance
predictor = None


def get_predictor():
    """Get or initialize the shape predictor"""
    global predictor
    if predictor is None:
        weights_dir = download_weights()  # This will download if needed
        predictor_path = os.path.join(
            weights_dir, "shape_predictor_68_face_landmarks.dat"
        )
        predictor = dlib.shape_predictor(predictor_path)
    return predictor


def get_face_landmarks(image: np.ndarray) -> Optional[dlib.full_object_detection]:
    """Detect face and get landmarks"""
    global predictor
    predictor = get_predictor()  # Get or initialize predictor
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return None
    return predictor(gray, faces[0])


def align_face(image: np.ndarray, landmarks: dlib.full_object_detection) -> np.ndarray:
    """Align face using eye landmarks"""
    left_eye = np.mean(
        [
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(37).x, landmarks.part(37).y),
            (landmarks.part(38).x, landmarks.part(38).y),
            (landmarks.part(39).x, landmarks.part(39).y),
            (landmarks.part(40).x, landmarks.part(40).y),
            (landmarks.part(41).x, landmarks.part(41).y),
        ],
        axis=0,
    )

    right_eye = np.mean(
        [
            (landmarks.part(42).x, landmarks.part(42).y),
            (landmarks.part(43).x, landmarks.part(43).y),
            (landmarks.part(44).x, landmarks.part(44).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(46).x, landmarks.part(46).y),
            (landmarks.part(47).x, landmarks.part(47).y),
        ],
        axis=0,
    )

    # Calculate angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate image
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return aligned


def crop_face(image: np.ndarray, landmarks: dlib.full_object_detection) -> np.ndarray:
    """Crop face using landmarks with padding"""
    # Get face bounds from landmarks
    x = [landmarks.part(i).x for i in range(68)]
    y = [landmarks.part(i).y for i in range(68)]

    x1, y1 = int(min(x)), int(min(y))
    x2, y2 = int(max(x)), int(max(y))

    # Add padding
    padding = 30
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image[y1:y2, x1:x2]


def preprocess_image(image: Union[str, np.ndarray]) -> torch.Tensor:
    """Preprocess image for model input
    Args:
        image: BGR image (HxWx3) or path to image
    Returns:
        torch.Tensor: Normalized tensor (1x3x128x128)
    """
    if isinstance(image, str):
        image = cv2.imread(image)  # This reads in BGR format
        if image is None:
            raise ValueError("Could not load image")

    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be either a path string or numpy array")

    # Ensure BGR format and uint8
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
    
    # Resize to 128x128
    image = cv2.resize(image, (128, 128))
    
    # Convert to float32
    image = image.astype(np.float32)
    
    # Normalize exactly as per original implementation
    image = (image - 127.5) / 128.0
    
    # Convert to tensor format (HWC -> CHW)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    img_tensor = torch.FloatTensor(image)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor
