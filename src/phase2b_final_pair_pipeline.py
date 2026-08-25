"""
=============================================================================
High-Resolution Real Pair Live Inspection Pipeline
=============================================================================
"""
import sys
import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
BEST_PT      = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
OUT_DIR      = PROJECT_ROOT / "output" / "phase2b_final_pair_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IP_CAMERA_URL = "http://10.113.196.111:8080/video"
IP_SHOT_URL   = "http://10.113.196.111:8080/shot.jpg"

def sep(title="", width=76, ch="="):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{ch*pad} {title} {ch*(width - pad - len(title) - 2)}")
    else:
        print(ch * width)

# Step 1: Capture High-Resolution Defective Frame from IP camera
print("Capturing High-Resolution Defective PCB Frame from live stream...")
frame_def = None
for url in [IP_SHOT_URL, IP_CAMERA_URL]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=4) as stream:
            if "shot.jpg" in url:
                img_bytes = stream.read()
                frame_def = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame_def is not None:
                    print(f"[SUCCESS] Captured full sensor frame from {url}: {frame_def.shape}")
                    break
            else:
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
                        frame_def = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame_def is not None:
                            print(f"[SUCCESS] Captured stream frame from {url}: {frame_def.shape}")
                            break
                if frame_def is not None:
                    break
    except Exception as e:
        print(f"[INFO] Endpoint {url} skipped: {e}")

if frame_def is None:
    raise RuntimeError("Could not capture frame from IP camera.")

# Save raw captured defective frame
def_raw_path = OUT_DIR / "1_real_defective_raw.jpg"
cv2.imwrite(str(def_raw_path), frame_def)

# Load the user's uploaded clean Good PCB image
good_src_path = PROJECT_ROOT / "output" / "phase2b_real_pair_experiment" / "1_real_good_pcb_raw.jpg"
frame_good = cv2.imread(str(good_src_path))
print(f"[LOADED] Real Good Master Frame: {frame_good.shape} <- {good_src_path}")

# Step 2: Accurate PCB ROI Extraction (HSV Green Mask)
def extract_pcb_roi(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([95, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 15000:
            x, y, w, h = cv2.boundingRect(c)
            if 0.5 < (w / float(h)) < 2.0:
                valid.append((c, area))
    if valid:
        best_c = max(valid, key=lambda x: x[1])[0]
        x, y, w, h = cv2.boundingRect(best_c)
        return frame[y:y+h, x:x+w]
    return frame

roi_good = extract_pcb_roi(frame_good)
roi_def  = extract_pcb_roi(frame_def)

cv2.imwrite(str(OUT_DIR / "2_roi_good.jpg"), roi_good)
cv2.imwrite(str(OUT_DIR / "2_roi_defective.jpg"), roi_def)
print(f"[ROI] Good ROI shape: {roi_good.shape}, Defective ROI shape: {roi_def.shape}")

# Resize both to standard 1024x1024 base
size = (1024, 1024)
roi_good_std = cv2.resize(roi_good, size, interpolation=cv2.INTER_AREA)
roi_def_std  = cv2.resize(roi_def, size, interpolation=cv2.INTER_AREA)

# Step 3: SIFT Homography Alignment (Rotation-invariant)
sift = cv2.SIFT_create(nfeatures=5000)
kp_g, des_g = sift.detectAndCompute(roi_good_std, None)
kp_d, des_d = sift.detectAndCompute(roi_def_std, None)

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = flann.knnMatch(des_d, des_g, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"[REGISTRATION] Robust SIFT feature matches: {len(good_matches)}")

if len(good_matches) >= 10:
    src_pts = np.float32([kp_d[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_g[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    inliers = int(np.sum(inlier_mask))
    print(f"[REGISTRATION] Homography inliers: {inliers}/{len(good_matches)} ({inliers/len(good_matches)*100:.1f}%)")
    aligned_def = cv2.warpPerspective(roi_def_std, H, size)
    
    match_img = cv2.drawMatches(roi_def_std, kp_d, roi_good_std, kp_g, good_matches[:60], None, flags=2)
    cv2.imwrite(str(OUT_DIR / "3_alignment_feature_matches.jpg"), match_img)
else:
    aligned_def = roi_def_std.copy()

aligned_good = roi_good_std.copy()
cv2.imwrite(str(OUT_DIR / "3_aligned_good.jpg"), aligned_good)
cv2.imwrite(str(OUT_DIR / "3_aligned_defective.jpg"), aligned_def)

# Step 4: Red Channel Extraction
red_good = aligned_good[:, :, 2]
red_def  = aligned_def[:, :, 2]
cv2.imwrite(str(OUT_DIR / "4_red_channel_good.jpg"), red_good)
cv2.imwrite(str(OUT_DIR / "4_red_channel_defective.jpg"), red_def)

# Step 5: Red-Channel Difference & Defect Candidate Extraction
diff_red_raw = cv2.absdiff(red_good, red_def)
cv2.imwrite(str(OUT_DIR / "5_raw_diff_red.jpg"), diff_red_raw)

# Threshold difference
_, thresh_red = cv2.threshold(diff_red_raw, 35, 255, cv2.THRESH_BINARY)
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
diff_clean = cv2.morphologyEx(thresh_red, cv2.MORPH_OPEN, k)
diff_clean = cv2.morphologyEx(diff_clean, cv2.MORPH_CLOSE, k)
cv2.imwrite(str(OUT_DIR / "6_morphological_candidate_regions.jpg"), diff_clean)

# Step 6: Identify Defect Bounding Boxes & Centroids
contours, _ = cv2.findContours(diff_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
candidate_regions = []
annotated_overlay = aligned_def.copy()

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area >= 25: # filter out sub-pixel noise
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

total_pixels = 1024 * 1024
total_diff_pixels = int(np.count_nonzero(diff_clean))
diff_area_pct = (total_diff_pixels / total_pixels) * 100.0

sep("HIGH-RESOLUTION REAL PAIR EXPERIMENT RESULTS")
print(f"Total Candidate Defect Regions Found : {len(candidate_regions)}")
print(f"Total Difference Area                : {total_diff_pixels} px ({diff_area_pct:.2f}% of board)")

print("\n--- Key Candidate Defect Regions ---")
# Sort by area descending
candidate_regions_sorted = sorted(candidate_regions, key=lambda x: x["area"], reverse=True)
for cand in candidate_regions_sorted[:10]:
    print(f"  Region D{cand['id']}: Area = {cand['area']} px, Centroid = {cand['centroid']}, BBox = {cand['bbox']}")

print(f"\nSaved all inspection stages to: {OUT_DIR}")
