import os
import gdown
from tqdm import tqdm

WEIGHT_URLS = {
    "shape_predictor_68_face_landmarks.dat": "https://drive.google.com/uc?id=17gmEzXQKU7Ushl2lw0JQ4nuDXCf43ME4",
    "LightCNN-V4_checkpoint.pth.tar": "https://drive.google.com/uc?id=1rLFXCIz3Mg-8zLnwa7X2_3u10Nolwgbo",
}

def download_weights() -> str:
    """Download model weights if they don't exist"""
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    for filename, url in WEIGHT_URLS.items():
        weight_path = os.path.join(weights_dir, filename)
        if not os.path.exists(weight_path):
            print(f"Downloading {filename}...")
            try:
                gdown.download(url, weight_path, quiet=False)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if os.path.exists(weight_path):
                    os.remove(weight_path)
                raise
    
    return weights_dir
