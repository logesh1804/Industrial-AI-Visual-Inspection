"""
Phase 2C — Scale-Aware Tiled YOLO Defect Detection Experiment.
Tests whether tiled inference improves real-camera defect detection compared to resizing the whole PCB.
"""
import sys
import os
import urllib.request
import time
from pathlib import Path
import cv2
import numpy as np
import csv
from ultralytics import YOLO

# Constants
PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
OUT_DIR = PROJECT_ROOT / "output" / "phase2c_tiled_experiment"
TILES_DIR = OUT_DIR / "tiles"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TILES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"

# Camera URL options
CAMERA_SOURCES = [
    {"type": "ip_shot", "url": "http://192.168.1.52:8080/shot.jpg"},
    {"type": "ip_video", "url": "http://192.168.1.52:8080/video"},
    {"type": "ip_shot", "url": "http://10.113.196.111:8080/shot.jpg"},
    {"type": "ip_video", "url": "http://10.113.196.111:8080/video"},
    {"type": "ip_shot", "url": "http://192.168.1.50:8080/shot.jpg"},
    {"type": "ip_video", "url": "http://192.168.1.50:8080/video"},
    {"type": "usb", "index": 1},
    {"type": "usb", "index": 2},
    {"type": "usb", "index": 0}
]

HOUGH_SETTINGS = {
    "minDist": 25,
    "param1": 50,
    "param2": 35,
    "minRadius": 6,
    "maxRadius": 30
}

def sep(title="", width=76, ch="="):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{ch*pad} {title} {ch*(width - pad - len(title) - 2)}")
    else:
        print(ch * width)

def order_points(pts):
    """Sort coordinates to: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def apply_nms(boxes, scores, classes, iou_threshold=0.45):
    """Class-specific non-maximum suppression in global coordinates"""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where((iou <= iou_threshold) | (classes[order[1:]] != classes[i]))[0]
        order = order[inds + 1]
        
    return keep

def main():
    sep("PHASE 2C TILED INFERENCE EXPERIMENT")
    
    # 1. Load YOLO model
    if not MODEL_PATH.exists():
        print(f"[ERROR] YOLO model not found at: {MODEL_PATH}")
        sys.exit(1)
    print(f"[LOADED] YOLO model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    
    # 2. Camera Input
    frame_raw = None
    connected_src = None
    
    print("Attempting to connect to camera...")
    for src in CAMERA_SOURCES:
        if src["type"] == "ip_shot":
            try:
                req = urllib.request.Request(src["url"])
                with urllib.request.urlopen(req, timeout=3) as stream:
                    img_bytes = stream.read()
                    frame_raw = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame_raw is not None:
                        connected_src = src["url"]
                        print(f"[SUCCESS] Connected to IP Camera (shot) {connected_src}: {frame_raw.shape}")
                        break
            except Exception as e:
                print(f"[INFO] IP shot failed for {src['url']}: {e}")
        elif src["type"] == "ip_video":
            try:
                req = urllib.request.Request(src["url"])
                with urllib.request.urlopen(req, timeout=3) as stream:
                    bytes_data = bytes()
                    # Try to read stream chunks
                    for _ in range(50):
                        chunk = stream.read(4096)
                        if not chunk:
                            break
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            frame_raw = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame_raw is not None:
                                connected_src = src["url"]
                                print(f"[SUCCESS] Connected to IP Camera (video) {connected_src}: {frame_raw.shape}")
                                break
                    if frame_raw is not None:
                        break
            except Exception as e:
                print(f"[INFO] IP video failed for {src['url']}: {e}")
        elif src["type"] == "usb":
            try:
                print(f"Trying USB camera index {src['index']}...")
                cap = cv2.VideoCapture(src["index"], cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(src["index"])
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret, frame_raw = cap.read()
                    cap.release()
                    if ret and frame_raw is not None:
                        connected_src = f"USB Camera Index {src['index']}"
                        print(f"[SUCCESS] Connected to USB camera: {frame_raw.shape}")
                        break
            except Exception as e:
                print(f"[INFO] USB camera {src['index']} failed: {e}")

    if frame_raw is None:
        print("[ERROR] Camera capture failed. All camera sources are currently unavailable.")
        sys.exit(1)
        
    # Save the original captured frame
    cv2.imwrite(str(OUT_DIR / "01_original_camera.jpg"), frame_raw)
    h_orig, w_orig = frame_raw.shape[:2]
    
    # 3. PCB ROI Detection
    print("\nDetecting PCB ROI...")
    hsv = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2HSV)
    # Mask green PCB board
    mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([95, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pcb_contour = None
    roi_bbox = None
    
    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 15000:
                x, y, w, h = cv2.boundingRect(c)
                if 0.4 < (w / float(h)) < 2.5:
                    valid_contours.append((c, area, (x, y, w, h)))
                    
        if valid_contours:
            best_tuple = max(valid_contours, key=lambda x: x[1])
            pcb_contour = best_tuple[0]
            roi_bbox = best_tuple[2]

    if pcb_contour is None or roi_bbox is None:
        print("[WARNING] Could not detect PCB contour. Using full image boundaries.")
        x, y, w, h = 0, 0, w_orig, h_orig
        roi_bbox = (x, y, w, h)
        # Create a mock rectangular contour
        pcb_contour = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
        
    x_r, y_r, w_r, h_r = roi_bbox
    roi_img = frame_raw[y_r:y_r+h_r, x_r:x_r+w_r]
    cv2.imwrite(str(OUT_DIR / "02_pcb_roi.jpg"), roi_img)
    print(f"[ROI] PCB ROI extracted: {roi_img.shape} at bbox [x={x_r}, y={y_r}, w={w_r}, h={h_r}]")

    # 4. Scale Estimation
    print("\nRunning Scale Estimation...")
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.medianBlur(gray_roi, 5)
    
    circles = cv2.HoughCircles(
        blurred_roi,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=HOUGH_SETTINGS["minDist"],
        param1=HOUGH_SETTINGS["param1"],
        param2=HOUGH_SETTINGS["param2"],
        minRadius=HOUGH_SETTINGS["minRadius"],
        maxRadius=HOUGH_SETTINGS["maxRadius"]
    )
    
    measured_pixel_diameter = None
    scale_reference_msg = "Scale calibration cannot yet be physically verified because no physical diameter was confirmed (continuing without scale normalization)."
    
    if circles is not None:
        circles_sorted = np.uint16(np.around(circles))
        # Take the largest circle as a potential screw hole candidate
        largest_circle = max(circles_sorted[0, :], key=lambda x: x[2])
        cx, cy, radius = largest_circle
        measured_pixel_diameter = int(radius * 2)
        
        # Save scale reference image
        scale_ref_img = roi_img.copy()
        cv2.circle(scale_ref_img, (cx, cy), radius, (0, 0, 255), 2)
        cv2.circle(scale_ref_img, (cx, cy), 2, (0, 255, 255), 3)
        cv2.putText(scale_ref_img, f"Dia: {measured_pixel_diameter}px", (cx - 40, cy - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT_DIR / "03_scale_reference.jpg"), scale_ref_img)
        print(f"[SCALE] Circular screw hole detected: Center=({cx}, {cy}), Radius={radius}px, Pixel Diameter={measured_pixel_diameter}px")
    else:
        print("[SCALE] No screw holes detected inside PCB ROI.")
        
    print(f"[SCALE] Status: {scale_reference_msg}")

    # 5. Perspective Correction / Alignment
    print("\nRunning Perspective Correction...")
    aligned_img = None
    peri = cv2.arcLength(pcb_contour, True)
    approx_poly = cv2.approxPolyDP(pcb_contour, 0.02 * peri, True)
    
    if len(approx_poly) == 4:
        rect = order_points(approx_poly.reshape(4, 2))
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        aligned_img = cv2.warpPerspective(frame_raw, M, (maxWidth, maxHeight))
        print(f"[ALIGN] Perspective corrected successfully: {aligned_img.shape}")
    else:
        print("[ALIGN] Corner detection failed to find exactly 4 points. Falling back to simple crop.")
        aligned_img = roi_img.copy()
        
    cv2.imwrite(str(OUT_DIR / "04_aligned_pcb.jpg"), aligned_img)
    h_align, w_align = aligned_img.shape[:2]

    # 6. Red Channel Extraction
    print("\nRunning Red Channel Extraction...")
    red_channel = aligned_img[:, :, 2]
    cv2.imwrite(str(OUT_DIR / "05_red_channel.jpg"), red_channel)
    print(f"[RED] Red channel extracted successfully.")

    # 7. Baseline Whole-PCB 640 Inference
    print("\nRunning Baseline Whole-PCB 640 Inference...")
    baseline_resized = cv2.resize(aligned_img, (640, 640), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_DIR / "06_baseline_whole_pcb_640.jpg"), baseline_resized)
    
    baseline_results = model.predict(source=baseline_resized, imgsz=640, conf=0.15, verbose=False)
    
    baseline_detections = []
    for r in baseline_results:
        names = r.names
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            baseline_detections.append({
                "class": names[cls],
                "confidence": conf,
                "bbox": [float(val) for val in xyxy]
            })
            
    print(f"[BASELINE] Detections found: {len(baseline_detections)}")
    for i, d in enumerate(baseline_detections):
        print(f"  Det {i+1}: {d['class']} (conf={d['confidence']:.2f}), bbox={d['bbox']}")

    # 8. Tiled Inference
    print("\nGenerating overlapping 640x640 tiles...")
    # Target tile parameters
    tile_size = 640
    overlap = 96
    stride = tile_size - overlap # 544
    
    # Pad image if it's smaller than 640 in height or width
    pad_h = max(tile_size, h_align)
    pad_w = max(tile_size, w_align)
    
    padded_red = np.zeros((pad_h, pad_w), dtype=np.uint8)
    padded_red[0:h_align, 0:w_align] = red_channel
    
    # Calculate tile coordinates
    y_coords = list(range(0, pad_h - tile_size + 1, stride))
    if y_coords[-1] != pad_h - tile_size:
        y_coords.append(pad_h - tile_size)
        
    x_coords = list(range(0, pad_w - tile_size + 1, stride))
    if x_coords[-1] != pad_w - tile_size:
        x_coords.append(pad_w - tile_size)
        
    # Generate tiles & visualization
    tile_visualization = cv2.cvtColor(padded_red, cv2.COLOR_GRAY2BGR)
    
    raw_tile_files = list(TILES_DIR.glob("*.jpg"))
    for file in raw_tile_files:
        try:
            file.unlink()
        except:
            pass
            
    tile_counter = 0
    tiled_detections_all = []
    
    for ty in y_coords:
        for tx in x_coords:
            tile_crop = padded_red[ty:ty+tile_size, tx:tx+tile_size]
            tile_name = f"tile_{tile_counter:03d}.jpg"
            tile_path = TILES_DIR / tile_name
            
            # Save raw tile
            cv2.imwrite(str(tile_path), tile_crop)
            
            # Draw boundary on grid visualization
            cv2.rectangle(tile_visualization, (tx, ty), (tx + tile_size, ty + tile_size), (0, 255, 0), 2)
            cv2.putText(tile_visualization, f"T{tile_counter}", (tx + 10, ty + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert tile to 3-channel for YOLO inference
            tile_3ch = cv2.cvtColor(tile_crop, cv2.COLOR_GRAY2BGR)
            tile_results = model.predict(source=tile_3ch, imgsz=640, conf=0.15, verbose=False)
            
            # Read and map coordinates back
            for r in tile_results:
                names = r.names
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    # Local coordinates
                    lx1, ly1, lx2, ly2 = xyxy
                    # Map to global coordinates
                    gx1 = tx + lx1
                    gy1 = ty + ly1
                    gx2 = tx + lx2
                    gy2 = ty + ly2
                    
                    tiled_detections_all.append({
                        "tile_id": tile_counter,
                        "class_id": cls,
                        "class_name": names[cls],
                        "confidence": conf,
                        "local_bbox": [float(val) for val in xyxy],
                        "global_bbox": [float(gx1), float(gy1), float(gx2), float(gy2)]
                    })
                    
            tile_counter += 1
            
    cv2.imwrite(str(OUT_DIR / "07_tile_grid_visualization.jpg"), tile_visualization)
    print(f"[TILES] Generated {tile_counter} tiles.")
    print(f"[TILES] Total raw detections found across all tiles: {len(tiled_detections_all)}")

    # 9. Duplicate Suppression / Global NMS
    print("\nRunning duplicate detection suppression (Global NMS)...")
    global_boxes = [d["global_bbox"] for d in tiled_detections_all]
    global_scores = [d["confidence"] for d in tiled_detections_all]
    global_classes = [d["class_id"] for d in tiled_detections_all]
    
    keep_indices = apply_nms(global_boxes, global_scores, global_classes, iou_threshold=0.45)
    final_tiled_detections = [tiled_detections_all[idx] for idx in keep_indices]
    
    print(f"[NMS] Suppressed {len(tiled_detections_all) - len(final_tiled_detections)} duplicates. Kept {len(final_tiled_detections)} final unique detections.")

    # 10. Reconstruct & Save Visualizations
    print("\nSaving final reconstructed visualization...")
    # Visualize final tiled detections on the complete aligned PCB
    tiled_annotated = cv2.cvtColor(padded_red, cv2.COLOR_GRAY2BGR)
    full_pcb_result = cv2.resize(aligned_img, (pad_w, pad_h), interpolation=cv2.INTER_CUBIC)
    
    for i, d in enumerate(final_tiled_detections):
        gx1, gy1, gx2, gy2 = map(int, d["global_bbox"])
        # Draw on tiled final binarized/red representation
        cv2.rectangle(tiled_annotated, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        cv2.putText(tiled_annotated, f"{d['class_name']} {d['confidence']:.2f}", (gx1, max(15, gy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
        # Draw on full PCB color aligned image
        cv2.rectangle(full_pcb_result, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        cv2.putText(full_pcb_result, f"{d['class_name']} {d['confidence']:.2f}", (gx1, max(15, gy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
    cv2.imwrite(str(OUT_DIR / "08_tiled_final_detections.jpg"), tiled_annotated)
    cv2.imwrite(str(OUT_DIR / "09_full_pcb_detection_result.jpg"), full_pcb_result)
    print(f"[RECONSTRUCT] Saved 08_tiled_final_detections.jpg and 09_full_pcb_detection_result.jpg")

    # 11. Write detections.csv
    csv_path = OUT_DIR / "detections.csv"
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tile_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        for d in final_tiled_detections:
            gx1, gy1, gx2, gy2 = d["global_bbox"]
            writer.writerow([d["tile_id"], d["class_id"], d["class_name"], f"{d['confidence']:.3f}", f"{gx1:.1f}", f"{gy1:.1f}", f"{gx2:.1f}", f"{gy2:.1f}"])
    print(f"[REPORT] Saved detections.csv containing final unique defects.")

    # 12. Evaluate Defect Scale details & Comparison
    baseline_count = len(baseline_detections)
    tiled_count = len(final_tiled_detections)
    
    # Compile comparison report text
    report_text = f"""==================================================
PHASE 2C COMPARISON REPORT
==================================================
Camera Source           : {connected_src}
Camera Capture size     : {w_orig}x{h_orig}
PCB ROI Bounding Box    : [x={x_r}, y={y_r}, w={w_r}, h={h_r}]
Aligned PCB Dimensions  : {w_align}x{h_align}
Screw Hole Detection    : {"Yes" if measured_pixel_diameter else "No"}
Measured Circle Diameter: {measured_pixel_diameter if measured_pixel_diameter else "N/A"} px
Scale px/mm             : Calibration cannot yet be physically verified (no physical size confirmed)
Generated Tiles count   : {tile_counter}
Tile Size / Overlap     : {tile_size} / {overlap}px (Stride = {stride}px)
YOLO Confidence threshold: 0.15
NMS IoU threshold       : 0.45

--------------------------------------------------
METHOD COMPARISON TABLE
--------------------------------------------------
Method        | Number of detections | High-confidence (>=0.50) | False Positives (Observed)
Whole PCB 640 | {baseline_count:<20} | {sum(1 for d in baseline_detections if d['confidence'] >= 0.50):<25} | None verified as ground truth
Tiled 640     | {tiled_count:<20} | {sum(1 for d in final_tiled_detections if d['confidence'] >= 0.50):<25} | None verified as ground truth

--------------------------------------------------
DEFECT-SCALE ANALYSIS EXAMPLE
--------------------------------------------------
Original Frame high-res board size: {w_align}x{h_align}
Standard Whole PCB resizing down to 640x640:
  Shrink factor: {w_align / 640.0:.2f}x
  Effect: A defect of size 12x8 pixels in high-res becomes only {12.0 / (w_align / 640.0):.1f}x{8.0 / (h_align / 640.0):.1f} pixels inside the 640x640 image, making it too blurred to detect.
Tiled strategy:
  Shrink factor: 1.0x (No resizing)
  Effect: The defect keeps its original 12x8 pixels size inside the 640x640 tile, preserving maximum resolution for detection.

--------------------------------------------------
VERDICT & RECOMMENDATIONS
--------------------------------------------------
Whether Tiling improved detection: {"Yes, it successfully preserved fine-scale trace information." if tiled_count > baseline_count else "Tiling completed successfully. Further ground truth testing required."}
Remaining Problems: B (Camera-domain mismatch), C (Silkscreen text false positives), and macro focus limits.
Recommendations: Collect custom defect annotations at this scale to retrain or fine-tune YOLO model for this specific camera domain.
"""
    # Save the comparison report file
    report_path = OUT_DIR / "comparison_report.txt"
    report_path.write_text(report_text)
    print(f"[REPORT] Saved comparison_report.txt successfully.")
    
    # Print the report output to console
    print("\n" + report_text)
    print("Phase 2C Tiled Inference Experiment completed successfully!")

if __name__ == "__main__":
    main()
