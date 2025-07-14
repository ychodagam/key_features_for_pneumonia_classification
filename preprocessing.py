# preprocessing.py
"""
Minimal preprocessing module for medical X-ray images:
- Resizes images to a consistent size.
- Normalizes pixel values to [0,1].

Usage:
  from pneumowave.preprocessing import preprocess_image, process_images_minimal

  # Single image
  processed_img = preprocess_image(img, target_size=(224,224))

  # Entire folder
  processed_images = process_images_minimal('/path/to/images', target_size=(224,224))
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def resize_image(image, target_size=(224, 224)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(image):
    normalized = image.astype(np.float32) / 255.0
    return normalized

def preprocess_image(image, target_size=None):
    if target_size is not None:
        image = resize_image(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image

def process_images_minimal(folder_path, target_size=(224,224), file_extensions=None):
    """
    Process all images from a folder with resizing and normalization.

    Parameters:
        folder_path (str): Path containing images.
        target_size (tuple): Desired output size.
        file_extensions (list, optional): Image extensions.

    Returns:
        list of tuples: (filename, preprocessed_image)
    """
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    image_files = []
    for ext in file_extensions:
        image_files = glob.glob(os.path.join(folder_path, f'*{ext}'))
        image_files += glob.glob(os.path.join(folder_path, f'*{ext.upper()}'))
    
    processed_images = []
    for file in tqdm(image_files, desc="Preprocessing Images (Minimal)"):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        processed = preprocess_image(img, target_size=target_size)
        processed_images.append((os.path.basename(file), processed))

    return processed_images
