import cv2
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Output log file
log_file = PROJECT_ROOT / "inspection_results.txt"

with open(log_file, "w") as f:
    f.write("=== PCB INSPECTION RESULTS ===\n\n")
    
    # 1. Inspect a dataset image
    train_dir = PROJECT_ROOT / "dataset" / "images" / "train"
    train_images = list(train_dir.glob("*.jpg"))
    if train_images:
        first_train = train_images[0]
        img_train = cv2.imread(str(first_train))
        f.write("--- Training Image (DeepPCB) ---\n")
        f.write(f"Path: {first_train.relative_to(PROJECT_ROOT)}\n")
        f.write(f"Shape: {img_train.shape}\n")
        f.write(f"Unique values (first row): {np.unique(img_train[0])}\n")
        f.write(f"Is binary check: min={img_train.min()}, max={img_train.max()}, mean={img_train.mean():.2f}\n\n")
    else:
        f.write("No training images found!\n\n")

    # 2. Inspect captured images
    capture_dir = PROJECT_ROOT / "captured_images"
    captured = list(capture_dir.glob("*.png"))
    if captured:
        f.write("--- Captured Images (Camera) ---\n")
        for cap_path in captured:
            img_cap = cv2.imread(str(cap_path))
            f.write(f"Path: {cap_path.relative_to(PROJECT_ROOT)}\n")
            f.write(f"Shape: {img_cap.shape}\n")
            f.write(f"Min={img_cap.min()}, Max={img_cap.max()}, Mean={img_cap.mean():.2f}\n")
            
            # Let's try basic PCB boundary detection using thresholding and contour finding
            # Convert to grayscale
            gray = cv2.cvtColor(img_cap, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Thresholding (Otsu)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            f.write(f"Found {len(contours)} contours.\n")
            
            # Filter contours by size to find the PCB
            large_contours = [c for c in contours if cv2.contourArea(c) > 10000]
            f.write(f"Contours > 10,000 area: {len(large_contours)}\n")
            if large_contours:
                # Find the largest one
                largest = max(large_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                f.write(f"Largest contour bounding box: x={x}, y={y}, w={w}, h={h} (Ratio={w/h:.2f})\n")
            f.write("-" * 30 + "\n")
    else:
        f.write("No captured images found!\n")

print("Image inspection script written.")
