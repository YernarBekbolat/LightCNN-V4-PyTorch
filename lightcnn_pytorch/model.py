import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from typing import Union, Tuple, Literal
import os
from .preprocessing import preprocess_image, get_face_landmarks, align_face
from .architectures import LightCNN_V4, LightCNN_29Layers_v2, resblock
from .utils import download_weights

class LightCNN:
    def __init__(self, device: str = None, **kwargs):
        """Initialize LightCNN model
        Args:
            device: str, optional - Device to run model on ('cuda' or 'cpu')
        Note:
            This version only supports LightCNN-V4
        """
        if 'model_name' in kwargs:
            print("Warning: model_name parameter is deprecated. Using LightCNN-V4.")
            
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        download_weights()
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

    def _load_model(self):
        model = LightCNN_V4()
        weights_file = "LightCNN-V4_checkpoint.pth.tar"
        weights_path = os.path.join(
            os.path.dirname(__file__), 
            "weights", 
            weights_file
        )

        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if "state_dict" in checkpoint:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    key = k.replace("module.", "")
                    if "fc2" not in key and "fc" not in key:
                        new_state_dict[key] = v
                model.load_state_dict(new_state_dict, strict=False)
            else:
                raise ValueError("Invalid checkpoint format")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise

        model.eval()
        return model

    def get_features(self, image: Union[str, np.ndarray], align: bool = True) -> np.ndarray:
        """Extract features from a face image"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError("Could not load image")

        if align:
            landmarks = get_face_landmarks(image)
            if landmarks is None:
                raise ValueError("No face detected in image")
            image = align_face(image, landmarks)

        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            self.model.eval()
            features = self.model(img_tensor)
            # Don't normalize here - let verify handle normalization

        return features.cpu().numpy().flatten()

    def verify(
        self,
        image1: Union[str, np.ndarray],
        image2: Union[str, np.ndarray],
        align: bool = True,
    ) -> Tuple[float, bool, float]:
        """Verify if two face images belong to the same person"""
        feat1 = self.get_features(image1, align=align)
        feat2 = self.get_features(image2, align=align)

        # Convert to numpy arrays if needed
        if isinstance(feat1, torch.Tensor):
            feat1 = feat1.cpu().numpy()
        if isinstance(feat2, torch.Tensor):
            feat2 = feat2.cpu().numpy()
        
        # Ensure numpy arrays
        feat1 = np.asarray(feat1, dtype=np.float32)
        feat2 = np.asarray(feat2, dtype=np.float32)

        # Normalize features (L2 norm)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 > 0:
            feat1 = feat1 / norm1
        if norm2 > 0:
            feat2 = feat2 / norm2
        
        # Calculate cosine similarity
        similarity = np.dot(feat1, feat2)
        
        # Use threshold from original implementation
        threshold = 0.7  # Adjusted based on original paper
        is_same = similarity > threshold

        # Calculate confidence (0-100%)
        confidence = (abs(similarity - threshold) / (1.0 - threshold)) * 100
        confidence = min(100.0, max(0.0, confidence))

        return similarity, is_same, confidence

# Global model cache
_models = {}

def get_model(device: str = None) -> LightCNN:
    global _models
    key = f"LightCNN-V4_{device}"
    if key not in _models:
        _models[key] = LightCNN(device=device)
    return _models[key]

def verify(
    image1: Union[str, np.ndarray], 
    image2: Union[str, np.ndarray], 
    device: str = None
) -> Tuple[float, bool, float]:
    model = get_model(device)
    return model.verify(image1, image2)

def get_features(
    image: Union[str, np.ndarray], 
    device: str = None
) -> np.ndarray:
    model = get_model(device)
    return model.get_features(image)
