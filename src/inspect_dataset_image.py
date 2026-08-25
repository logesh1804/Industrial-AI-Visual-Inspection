import cv2
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
train_dir = PROJECT_ROOT / "dataset" / "images" / "train"
images = list(train_dir.glob("*.jpg"))

output_file = PROJECT_ROOT / "dataset_info.txt"

with open(output_file, "w") as f:
    if not images:
        f.write("No images found in dataset/images/train\n")
    else:
        first_img_path = images[0]
        img = cv2.imread(str(first_img_path))
        f.write(f"Image name: {first_img_path.name}\n")
        f.write(f"Shape: {img.shape}\n")
        f.write(f"Dtype: {img.dtype}\n")
        f.write(f"Min value: {img.min()}, Max value: {img.max()}\n")
        
        # Check if it is grayscale (if all 3 channels are identical)
        if len(img.shape) == 3 and img.shape[2] == 3:
            b, g, r = cv2.split(img)
            if (b == g).all() and (g == r).all():
                f.write("Image is visually grayscale (all channels are identical).\n")
            else:
                f.write("Image is color (channels are different).\n")
        else:
            f.write("Image is single channel.\n")
            
        # Let's count how many images are there
        f.write(f"Total training images: {len(images)}\n")

print("Inspection completed.")
