"""
=============================================================================
Phase 2B: Real Pair (Real Good vs Real Defective) Camera Experiment
=============================================================================
Performs genuine dual-frame capture and structural difference without any
artificial inpainting/repair:
  1. Capture Real Good PCB Frame from IP Camera.
  2. Capture Real Defective PCB Frame from IP Camera.
  3. Extract PCB ROI from both frames.
  4. Perform Homography Alignment (register Defective to Good frame).
  5. Compute Structural Difference Image: |Good_Aligned - Defective_Aligned|
  6. Verify detection at:
     - A. Open circuit (at the actual GREEN CIRCLE cut location)
     - B. Mousebite (at actual mousebite notch)
     - C. Solder short (at actual solder bridge)
  7. Compute difference areas, bounding boxes, overlap & false differences.
=============================================================================
"""

import sys
import os
import urllib.request
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PT      = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
OUT_DIR      = PROJECT_ROOT / "output" / "phase2b_real_pair_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IP_CAMERA_URL = "http://192.168.1.50:8080/video"

CLASSES = {0: "open", 1: "short", 2: "mousebite", 3: "spur", 4: "spurious_copper", 5: "pin_hole"}

# Canonical coordinates on 1024x1024 aligned frame
# Open circuit is at [x=485..530, y=470..500] (the user's green circle location)
# Mousebite is at [x=540..595, y=555..610]
# Solder short is at [x=710..790, y=460..635]
GT_DEFECT_REGIONS = {
    "open_circuit": {"name": "Open Circuit (Green Circle)", "box": [485, 470, 530, 500]},
    "mousebite":    {"name": "Mousebite (Trace Notch)",     "box": [540, 555, 595, 610]},
    "solder_short": {"name": "Solder Short (Bridge)",       "box": [710, 460, 790, 635]}
}

def sep(title="", width=76, ch="="):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{ch*pad} {title} {ch*(width - pad - len(title) - 2)}")
    else:
        print(ch * width)

def grab_ip_frame(timeout=4):
    req = urllib.request.Request(IP_CAMERA_URL)
    with urllib.request.urlopen(req, timeout=timeout) as stream:
        bytes_data = bytes()
        for _ in range(40):
            chunk = stream.read(4096)
            if not chunk:
                break
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
    raise RuntimeError("Failed to grab frame from IP Camera.")

def extract_pcb_roi_and_align_pair(raw_good, raw_defective):
    """
    Extracts PCB ROI from both frames and registers the Defective PCB
    to the Good PCB coordinate system using Feature Matching (ORB + Homography).
    """
    def get_roi(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return frame[y:y+h, x:x+w]
        return frame

    roi_good = get_roi(raw_good)
    roi_def  = get_roi(raw_defective)
    
    # Resize both to standard 1024x1024 base
    roi_good_1024 = cv2.resize(roi_good, (1024, 1024), interpolation=cv2.INTER_AREA)
    roi_def_1024  = cv2.resize(roi_def, (1024, 1024), interpolation=cv2.INTER_AREA)
    
    # Feature-based sub-pixel alignment using ORB
    orb = cv2.ORB_create(nfeatures=3000)
    kp_g, des_g = orb.detectAndCompute(roi_good_1024, None)
    kp_d, des_d = orb.detectAndCompute(roi_def_1024, None)
    
    aligned_def = roi_def_1024.copy()
    if des_g is not None and des_d is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des_d, des_g), key=lambda x: x.distance)
        good_matches = matches[:min(len(matches), 200)]
        if len(good_matches) >= 12:
            src_pts = np.float32([kp_d[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_g[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            if H is not None:
                aligned_def = cv2.warpPerspective(roi_def_1024, H, (1024, 1024))
                
    return roi_good_1024, aligned_def

def evaluate_structural_difference(good_aligned, def_aligned):
    """
    Computes genuine optical structural difference between the real Good PCB
    and real Defective PCB (Red Channel difference to maximize copper contrast).
    """
    red_good = good_aligned[:, :, 2]
    red_def  = def_aligned[:, :, 2]
    
    # Absolute difference on Red channel
    diff_abs = cv2.absdiff(red_good, red_def)
    
    # Threshold at 35 intensity difference
    _, diff_bin = cv2.threshold(diff_abs, 35, 255, cv2.THRESH_BINARY)
    
    # Clean small sub-pixel registration jitter with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_clean = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel)
    
    total_diff_pixels = int(np.count_nonzero(diff_clean))
    total_board_pixels = 1024 * 1024
    diff_area_pct = (total_diff_pixels / total_board_pixels) * 100.0
    
    # Find candidate difference contours & bounding boxes
    contours, _ = cv2.findContours(diff_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes = []
    for c in contours:
        if cv2.contourArea(c) >= 15: # Filter out tiny noise specks (<15 px)
            x, y, w, h = cv2.boundingRect(c)
            candidate_boxes.append([x, y, x + w, y + h, int(cv2.contourArea(c))])
            
    # Check overlap against each known ground-truth defect
    defect_checks = {}
    for dkey, gt in GT_DEFECT_REGIONS.items():
        gx1, gy1, gx2, gy2 = gt["box"]
        # Check if any candidate box overlaps or is inside the GT region
        overlap_found = False
        max_overlap_area = 0
        for cb in candidate_boxes:
            cx1, cy1, cx2, cy2, carea = cb
            # Overlap coordinates
            ox1 = max(gx1, cx1)
            oy1 = max(gy1, cy1)
            ox2 = min(gx2, cx2)
            oy2 = min(gy2, cy2)
            if ox1 < ox2 and oy1 < oy2:
                overlap_found = True
                max_overlap_area += (ox2 - ox1) * (oy2 - oy1)
                
        defect_checks[dkey] = {
            "name": gt["name"],
            "detected_in_diff": overlap_found,
            "diff_pixels_in_region": max_overlap_area
        }
        
    return diff_clean, candidate_boxes, diff_area_pct, defect_checks

if __name__ == "__main__":
    sep("PHASE 2B: REAL PAIR CAMERA EXPERIMENT (GENUINE GOOD vs DEFECTIVE)")
    print("Script initialized and ready to run.")
