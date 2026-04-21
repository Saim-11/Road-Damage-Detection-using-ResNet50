"""
Utility functions for road damage detection project
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

# Setup logging
def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Image loading and processing
def load_image(image_path: str) -> np.ndarray:
    """Load image from file"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_label(label_path: str) -> np.ndarray:
    """Load YOLO format labels"""
    if not os.path.exists(label_path):
        return np.array([])
    
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                annotations.append([float(x) for x in parts])
    
    return np.array(annotations) if annotations else np.array([])

def save_visualization(image: np.ndarray, title: str, output_dir: str, filename: str):
    """Save visualization image"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(filepath, image)

# Metrics computation
def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index"""
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    C1 = 6.5025
    C2 = 58.5225
    
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel = kernel @ kernel.T
    
    mu1 = cv2.filter2D(img1, -1, kernel)
    mu2 = cv2.filter2D(img2, -1, kernel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

# YOLO format conversions
def xyxy_to_yolo(bbox: Tuple, img_width: int, img_height: int) -> Tuple:
    """Convert xyxy format to YOLO format (x_center, y_center, width, height) normalized"""
    x1, y1, x2, y2 = bbox
    
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return (x_center, y_center, width, height)

def yolo_to_xyxy(bbox: Tuple, img_width: int, img_height: int) -> Tuple:
    """Convert YOLO format to xyxy format"""
    x_center, y_center, width, height = bbox
    
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return (max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2))

def draw_bboxes(image: np.ndarray, annotations: np.ndarray, img_height: int, img_width: int, class_names: Dict = None):
    """Draw bounding boxes on image"""
    img_copy = image.copy()
    
    for ann in annotations:
        class_id = int(ann[0])
        bbox = ann[1:5]
        x1, y1, x2, y2 = yolo_to_xyxy(bbox, img_width, img_height)
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw class label
        if class_names and class_id in class_names:
            label = class_names[class_id]
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_copy

# Directory utilities
def get_image_files(directory: str, extensions: List[str] = ['.jpg', '.png', '.jpeg']) -> List[str]:
    """Get all image files from directory"""
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))
        image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted([str(f) for f in image_files])

def create_splits_summary(data_dir: str, output_file: str):
    """Create summary of dataset splits"""
    splits = ['train', 'val', 'test']
    summary = {}
    
    for split in splits:
        split_dir = os.path.join(data_dir, split, 'images')
        if os.path.exists(split_dir):
            count = len(get_image_files(split_dir))
            summary[split] = count
    
    total = sum(summary.values())
    
    with open(output_file, 'w') as f:
        f.write("Dataset Split Summary\n")
        f.write("=" * 40 + "\n\n")
        for split, count in summary.items():
            percentage = (count / total) * 100
            f.write(f"{split.capitalize():10} : {count:5} images ({percentage:5.1f}%)\n")
        f.write(f"\n{'Total':10} : {total:5} images\n")
    
    print(f"✓ Split summary saved to {output_file}")
    for split, count in summary.items():
        print(f"  {split}: {count} images")

print("Utilities loaded")
