import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
img_path = PROJECT_ROOT / "test_images" / "pcb_color_defected.jpg"
output_dir = PROJECT_ROOT / "output" / "binarization_tests"
output_dir.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(img_path))
if img is None:
    print(f"Failed to load image at {img_path}")
    exit(1)

# Method 1: Otsu on standard Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, otsu_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Ensure mostly white substrate (Otsu background is usually white, if not invert)
if np.sum(otsu_gray == 255) < otsu_gray.size * 0.5:
    otsu_gray = cv2.bitwise_not(otsu_gray)
cv2.imwrite(str(output_dir / "1_otsu_grayscale.png"), otsu_gray)

# Method 2: Otsu on Red Channel (Green mask is dark, copper/solder is bright)
red = img[:, :, 2]
_, otsu_red = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if np.sum(otsu_red == 255) < otsu_red.size * 0.5:
    otsu_red = cv2.bitwise_not(otsu_red)
cv2.imwrite(str(output_dir / "2_otsu_red.png"), otsu_red)

# Method 3: Adaptive thresholding with a very large block size (e.g. 101) to keep traces solid
# (Standard grayscale)
adaptive_large = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    101,  # Large block size to avoid hollow centers
    8
)
if np.sum(adaptive_large == 255) < adaptive_large.size * 0.5:
    adaptive_large = cv2.bitwise_not(adaptive_large)
cv2.imwrite(str(output_dir / "3_adaptive_large.png"), adaptive_large)

# Method 4: Adaptive thresholding on Red Channel with block size 101
adaptive_red_large = cv2.adaptiveThreshold(
    red,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    101,
    8
)
if np.sum(adaptive_red_large == 255) < adaptive_red_large.size * 0.5:
    adaptive_red_large = cv2.bitwise_not(adaptive_red_large)
cv2.imwrite(str(output_dir / "4_adaptive_red_large.png"), adaptive_red_large)

print("Binarization test images generated in output/binarization_tests/")
