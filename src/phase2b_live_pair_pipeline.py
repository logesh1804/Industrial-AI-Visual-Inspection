"""
=============================================================================
Phase 2B: Full Live Real Pair (Real Good vs Real Defective) Evaluation Pipeline
=============================================================================
"""
import sys
import os
import urllib.request
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
OUT_DIR = PROJECT_ROOT / "output" / "phase2b_real_pair_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IP_CAMERA_URL = "http://10.113.196.111:8080/video"

def sep(title="", width=76, ch="="):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{ch*pad} {title} {ch*(width - pad - len(title) - 2)}")
    else:
        print(ch * width)

# Step 1: Capture Defective Frame
print("Capturing Defective PCB Frame from live stream...")
req = urllib.request.Request(IP_CAMERA_URL)
with urllib.request.urlopen(req, timeout=4) as stream:
    bytes_data = bytes()
    for _ in range(50):
        chunk = stream.read(4096)
        if not chunk:
            break
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame_def = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame_def is not None:
                def_path = OUT_DIR / "1_real_defective_pcb_raw.jpg"
                cv2.imwrite(str(def_path), frame_def)
                print(f"[SUCCESS] Real Defective PCB Captured: {frame_def.shape} -> {def_path}")
                break

# Load Good PCB Frame
good_path = OUT_DIR / "1_real_good_pcb_raw.jpg"
frame_good = cv2.imread(str(good_path))
print(f"[SUCCESS] Real Good PCB Loaded: {frame_good.shape} <- {good_path}")

# Step 2: PCB ROI Extraction
def extract_pcb_roi(frame, debug_name=""):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Mask green PCB board
    mask = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([95, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, None
    h, w = frame.shape[:2]
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10000:
            x, y, cw, ch = cv2.boundingRect(c)
            if y < h * 0.7: # Ignore bottom clamp/fixture
                valid_contours.append((c, area))
    if valid_contours:
        best_c = max(valid_contours, key=lambda item: item[1])[0]
        x, y, cw, ch = cv2.boundingRect(best_c)
        roi = frame[y:y+ch, x:x+cw]
        return roi, best_c
    return frame, None

roi_good, cont_good = extract_pcb_roi(frame_good, "good")
roi_def, cont_def   = extract_pcb_roi(frame_def, "defective")

cv2.imwrite(str(OUT_DIR / "2_roi_good.jpg"), roi_good)
cv2.imwrite(str(OUT_DIR / "2_roi_defective.jpg"), roi_def)
print(f"[ROI] Good ROI shape: {roi_good.shape}, Defective ROI shape: {roi_def.shape}")

# Resize both to standard 1024x1024 base for registration
size = (1024, 1024)
roi_good_std = cv2.resize(roi_good, size, interpolation=cv2.INTER_AREA)
roi_def_std  = cv2.resize(roi_def, size, interpolation=cv2.INTER_AREA)

# Step 3: Homography Alignment & Registration (SIFT/ORB)
sift = cv2.SIFT_create(nfeatures=4000)
kp_g, des_g = sift.detectAndCompute(roi_good_std, None)
kp_d, des_d = sift.detectAndCompute(roi_def_std, None)

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = flann.knnMatch(des_d, des_g, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"[REGISTRATION] Number of robust feature matches: {len(good_matches)}")

if len(good_matches) >= 10:
    src_pts = np.float32([kp_d[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_g[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    inliers = int(np.sum(inlier_mask))
    print(f"[REGISTRATION] Homography inliers: {inliers}/{len(good_matches)} ({inliers/len(good_matches)*100:.1f}%)")
    aligned_def = cv2.warpPerspective(roi_def_std, H, size)
    
    match_img = cv2.drawMatches(roi_def_std, kp_d, roi_good_std, kp_g, good_matches[:50], None, flags=2)
    cv2.imwrite(str(OUT_DIR / "3_alignment_feature_matches.jpg"), match_img)
else:
    print("[WARNING] Insufficient feature matches. Using direct resize.")
    aligned_def = roi_def_std.copy()

aligned_good = roi_good_std.copy()
cv2.imwrite(str(OUT_DIR / "3_aligned_good.jpg"), aligned_good)
cv2.imwrite(str(OUT_DIR / "3_aligned_defective.jpg"), aligned_def)

# Step 4: Red Channel Extraction
red_good = aligned_good[:, :, 2]
red_def  = aligned_def[:, :, 2]
cv2.imwrite(str(OUT_DIR / "4_red_channel_good.jpg"), red_good)
cv2.imwrite(str(OUT_DIR / "4_red_channel_defective.jpg"), red_def)

# Grayscale Extraction for comparison
gray_good = cv2.cvtColor(aligned_good, cv2.COLOR_BGR2GRAY)
gray_def  = cv2.cvtColor(aligned_def, cv2.COLOR_BGR2GRAY)

# Step 5: Raw & Thresholded Difference Computations
diff_red_raw  = cv2.absdiff(red_good, red_def)
diff_gray_raw = cv2.absdiff(gray_good, gray_def)
diff_rgb_raw  = cv2.absdiff(aligned_good, aligned_def)

cv2.imwrite(str(OUT_DIR / "5_raw_diff_red.jpg"), diff_red_raw)
cv2.imwrite(str(OUT_DIR / "5_raw_diff_gray.jpg"), diff_gray_raw)

# Threshold difference (intensity delta > 30)
_, thresh_red = cv2.threshold(diff_red_raw, 30, 255, cv2.THRESH_BINARY)
_, thresh_gray = cv2.threshold(diff_gray_raw, 30, 255, cv2.THRESH_BINARY)

# Morphological cleanup
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
diff_clean = cv2.morphologyEx(thresh_red, cv2.MORPH_OPEN, k)
diff_clean = cv2.morphologyEx(diff_clean, cv2.MORPH_CLOSE, k)

cv2.imwrite(str(OUT_DIR / "6_thresholded_diff_red.jpg"), thresh_red)
cv2.imwrite(str(OUT_DIR / "6_morphological_candidate_regions.jpg"), diff_clean)

# Step 6: Candidate Regions, Bounding Boxes, Areas & Centroids
contours, _ = cv2.findContours(diff_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
candidate_regions = []
annotated_overlay = aligned_def.copy()

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area >= 20: # filter out sub-pixel jitter < 20 px
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        cx = int(M["m10"] / (M["m00"] + 1e-5))
        cy = int(M["m01"] / (M["m00"] + 1e-5))
        candidate_regions.append({
            "id": i + 1,
            "bbox": [x, y, x + w, y + h],
            "area": int(area),
            "centroid": (cx, cy)
        })
        cv2.rectangle(annotated_overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(annotated_overlay, (cx, cy), 3, (0, 255, 255), -1)
        cv2.putText(annotated_overlay, f"D{i+1}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite(str(OUT_DIR / "7_candidate_bounding_boxes_overlay.jpg"), annotated_overlay)

# Step 7: Quantitative Analysis & Metrics
total_pixels = 1024 * 1024
total_diff_pixels = int(np.count_nonzero(diff_clean))
diff_area_pct = (total_diff_pixels / total_pixels) * 100.0

mask_cand = np.zeros((1024, 1024), dtype=np.uint8)
for cand in candidate_regions:
    x1, y1, x2, y2 = cand["bbox"]
    mask_cand[y1:y2, x1:x2] = 255

signal_pixels = diff_red_raw[mask_cand == 255]
bg_pixels     = diff_red_raw[mask_cand == 0]

mean_signal = float(np.mean(signal_pixels)) if len(signal_pixels) > 0 else 0.0
mean_bg     = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 1.0
snr = mean_signal / (mean_bg + 1e-5)

sep("PHASE 2B: REAL PAIR EXPERIMENT RESULTS")
print(f"Total Candidate Defect Regions Found : {len(candidate_regions)}")
print(f"Total Difference Area                : {total_diff_pixels} px ({diff_area_pct:.2f}% of board)")
print(f"Signal-to-Background Ratio (Red Ch)  : {snr:.2f}x (Mean Signal: {mean_signal:.1f}, Mean BG: {mean_bg:.1f})")

print("\n--- Candidate Anomaly Regions List ---")
for cand in candidate_regions[:10]:
    print(f"  Region D{cand['id']}: Area = {cand['area']} px, Centroid = {cand['centroid']}, BBox = {cand['bbox']}")
if len(candidate_regions) > 10:
    print(f"  ... and {len(candidate_regions) - 10} more smaller anomaly clusters.")

print("\nSaved all visual stages to:", OUT_DIR)
