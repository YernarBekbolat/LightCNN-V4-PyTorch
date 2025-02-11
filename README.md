# LightCNN

Fast and accurate face recognition model optimized for Central Asian faces. This PyTorch implementation provides an easy-to-use interface for face verification and feature extraction.

## Features
- LightCNN-V4 architecture optimized for Central Asian faces
- Fast and lightweight face verification
- Automatic face alignment and preprocessing
- Easy-to-use Python API
- Pre-trained model weights included

## Installation
```bash
pip install lightcnn-pytorch
```

## Quick Start
```python
from lightcnn_pytorch import LightCNN

# Initialize model (automatically downloads weights)
model = LightCNN()

# Verify two face images
similarity, is_same, confidence = model.verify("person1.jpg", "person2.jpg")
print(f"Similarity score: {similarity:.3f}")
print(f"Same person: {is_same}")
print(f"Confidence: {confidence:.3f}%")
```

## Usage

### Face Verification
```python
# Compare two face images
similarity, is_same, confidence = model.verify(image1, image2)
```

The `verify` method returns:
- `similarity`: Cosine similarity score (-1 to 1)
- `is_same`: Boolean indicating if images are of the same person
- `confidence`: Confidence score (0-100%)

### Feature Extraction
```python
# Get face embedding features
features = model.get_features(image)
```

### Input Formats
- File path (`str`)
- BGR image array (`numpy.ndarray`)
- RGB image array will be converted to BGR automatically

### Device Selection
```python
# Use specific device
model = LightCNN(device="cuda")  # or "cpu"
```

## Model Details
- **Architecture**: LightCNN-V4
- **Input**: 128x128 BGR image
- **Preprocessing**: `(x - 127.5) / 128.0`
- **Output**: 256-dimensional feature vector
- **Verification threshold**: 0.7



