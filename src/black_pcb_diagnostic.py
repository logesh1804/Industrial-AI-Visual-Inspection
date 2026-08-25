"""
Black PCB Controlled Diagnostic Experiment
Tests 3 Preprocessing methods (CLAHE, Simple Grayscale, Original BGR)
across 4 Confidence levels (0.10, 0.20, 0.30, 0.45) = 12 Tests.
"""
import sys
import os
import shutil
import csv
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
BASE_OUT_DIR = PROJECT_ROOT / "output" / "phase2c_1_4hole_dynamic_tiled"
DIAG_OUT_DIR = BASE_OUT_DIR / "black_pcb_diagnostic"
DIAG_OUT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = Path(r"C:\Users\sabarishclean\.gemini\antigravity-ide\brain\e1001de1-7b9a-49a0-b0fb-c022f925ab3c")
ART_DIAG_DIR = ARTIFACTS_DIR / "black_pcb_diagnostic"
ART_DIAG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
INPUT_IMG_PATH = BASE_OUT_DIR / "01_original_camera.jpg"

IOU_THRESHOLD = 0.45
CONF_LEVELS = [0.10, 0.20, 0.30, 0.45]
PREPROCESSING_TYPES = ["CLAHE", "GRAY", "BGR"]

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def apply_nms(boxes, scores, classes, iou_threshold=0.45):
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

def extract_universal_pcb_roi(frame):
    h_orig, w_orig = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    mask_green = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([90, 255, 255]))
    mask_blue  = cv2.inRange(hsv, np.array([95, 35, 35]), np.array([140, 255, 255]))
    mask_red1  = cv2.inRange(hsv, np.array([0, 40, 40]),  np.array([12, 255, 255]))
    mask_red2  = cv2.inRange(hsv, np.array([165, 40, 40]),np.array([180, 255, 255]))
    mask_red   = mask_red1 | mask_red2
    mask_yellow = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([30, 255, 255]))
    _, mask_black = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    
    combined_mask = mask_green | mask_blue | mask_red | mask_yellow | mask_black
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_c = None
    best_bbox = None
    detected_color = "Black"
    
    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 10000:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h)
                if 0.35 < aspect < 2.8 and (w < w_orig * 0.98 or h < h_orig * 0.98):
                    valid_contours.append((c, area, (x, y, w, h)))
        if valid_contours:
            best_tuple = max(valid_contours, key=lambda x: x[1])
            best_c = best_tuple[0]
            best_bbox = best_tuple[2]
            
    if best_c is None or best_bbox is None:
        best_bbox = (0, 0, w_orig, h_orig)
        best_c = np.array([[[0, 0]], [[w_orig, 0]], [[w_orig, h_orig]], [[0, h_orig]]], dtype=np.int32)
        
    return best_c, best_bbox, detected_color

def detect_silkscreen_and_text_regions(aligned_img):
    h, w = aligned_img.shape[:2]
    gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    
    kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_th)
    _, thresh_bright = cv2.threshold(tophat, 35, 255, cv2.THRESH_BINARY)
    
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_th)
    _, thresh_dark = cv2.threshold(blackhat, 35, 255, cv2.THRESH_BINARY)
    
    silk_raw = thresh_bright | thresh_dark
    
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        for p in regions:
            hull = cv2.convexHull(p.reshape(-1, 1, 2))
            cv2.drawContours(silk_raw, [hull], -1, 255, -1)
    except:
        pass
        
    kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
    word_clustered = cv2.morphologyEx(silk_raw, cv2.MORPH_CLOSE, kernel_word)
    word_clustered = cv2.dilate(word_clustered, kernel_word, iterations=1)
    
    contours, _ = cv2.findContours(word_clustered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_boxes = []
    margin = 8
    for c in contours:
        area = cv2.contourArea(c)
        if 80 < area < (w * h * 0.15):
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < 150 and bw < 500:
                tx1 = max(0, x - margin)
                ty1 = max(0, y - margin)
                tx2 = min(w, x + bw + margin)
                ty2 = min(h, y + bh + margin)
                text_boxes.append([tx1, ty1, tx2, ty2])
                
    blurred = cv2.medianBlur(gray, 5)
    vias = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=28, minRadius=6, maxRadius=25
    )
    regular_vias = []
    if vias is not None:
        vias = np.int32(np.around(vias))[0, :]
        for v in vias:
            regular_vias.append((int(v[0]), int(v[1]), int(v[2])))
            
    return text_boxes, regular_vias

def filter_defects_with_text_and_pads(raw_detections, text_boxes, regular_vias):
    valid_defects = []
    negated_defects = []
    
    for d in raw_detections:
        gx1, gy1, gx2, gy2 = d["global_bbox"]
        cx = (gx1 + gx2) / 2.0
        cy = (gy1 + gy2) / 2.0
        bw = gx2 - gx1
        bh = gy2 - gy1
        det_area = bw * bh
        cname = d["class_name"]
        
        is_negated = False
        negate_reason = ""
        
        for tb in text_boxes:
            tx1, ty1, tx2, ty2 = tb
            if (tx1 <= cx <= tx2) and (ty1 <= cy <= ty2):
                is_negated = True
                negate_reason = "Silkscreen/Text Region"
                break
                
            ix1 = max(gx1, tx1)
            iy1 = max(gy1, ty1)
            ix2 = min(gx2, tx2)
            iy2 = min(gy2, ty2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if det_area > 0 and (inter_area / float(det_area)) > 0.25:
                    is_negated = True
                    negate_reason = "Silkscreen/Text Overlap"
                    break
                    
        if not is_negated and cname == "pin_hole":
            for vx, vy, vr in regular_vias:
                dist_sq = (cx - vx)**2 + (cy - vy)**2
                if dist_sq <= (vr + 8)**2:
                    is_negated = True
                    negate_reason = "Regular Solder Via/Pad"
                    break
                    
        d_copy = dict(d)
        if is_negated:
            d_copy["negated"] = True
            d_copy["negate_reason"] = negate_reason
            negated_defects.append(d_copy)
        else:
            d_copy["negated"] = False
            valid_defects.append(d_copy)
            
    return valid_defects, negated_defects

def run_diagnostic():
    print("==================================================")
    print("RUNNING BLACK PCB CONTROLLED DIAGNOSTIC EXPERIMENT")
    print("==================================================")
    
    if not INPUT_IMG_PATH.exists():
        print(f"[ERROR] Camera image not found at {INPUT_IMG_PATH}")
        sys.exit(1)
    frame_captured = cv2.imread(str(INPUT_IMG_PATH))
    h_orig, w_orig = frame_captured.shape[:2]
    print(f"[LOAD] Original frame: {w_orig}x{h_orig}")
    
    # 1. ROI & Registration
    pcb_contour, (rx, ry, rw, rh), detected_color = extract_universal_pcb_roi(frame_captured)
    roi_img = frame_captured[ry:ry+rh, rx:rx+rw]
    
    # Registration fallback matching production
    peri = cv2.arcLength(pcb_contour, True)
    approx_poly = cv2.approxPolyDP(pcb_contour, 0.02 * peri, True)
    if len(approx_poly) == 4:
        rect = order_points(approx_poly.reshape(4, 2))
        (tl_pt, tr_pt, br_pt, bl_pt) = rect
        widthA = np.sqrt(((br_pt[0] - bl_pt[0]) ** 2) + ((br_pt[1] - bl_pt[1]) ** 2))
        widthB = np.sqrt(((tr_pt[0] - tl_pt[0]) ** 2) + ((tr_pt[1] - tl_pt[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr_pt[0] - br_pt[0]) ** 2) + ((tr_pt[1] - br_pt[1]) ** 2))
        heightB = np.sqrt(((tl_pt[0] - bl_pt[0]) ** 2) + ((tl_pt[1] - bl_pt[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst_pts = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst_pts)
        aligned_img = cv2.warpPerspective(frame_captured, M, (maxWidth, maxHeight))
    else:
        aligned_img = roi_img.copy()
        
    h_align, w_align = aligned_img.shape[:2]
    print(f"[ALIGN] Registered PCB dimensions: {w_align}x{h_align}")
    
    # 2. Silkscreen Text & Pads
    text_boxes, regular_vias = detect_silkscreen_and_text_regions(aligned_img)
    print(f"[TEXT ENGINE] Detected {len(text_boxes)} text zones, {len(regular_vias)} regular vias")
    
    # 3. Dynamic Tiling Parameters
    tile_size = 640
    overlap = 96
    stride = tile_size - overlap # 544
    
    pad_h = max(tile_size, h_align)
    pad_w = max(tile_size, w_align)
    
    y_coords = []
    y = 0
    while y + tile_size <= pad_h:
        y_coords.append(y)
        y += stride
    if pad_h > tile_size and (len(y_coords) == 0 or y_coords[-1] != pad_h - tile_size):
        y_coords.append(pad_h - tile_size)
    if len(y_coords) == 0:
        y_coords = [0]
        
    x_coords = []
    x = 0
    while x + tile_size <= pad_w:
        x_coords.append(x)
        x += stride
    if pad_w > tile_size and (len(x_coords) == 0 or x_coords[-1] != pad_w - tile_size):
        x_coords.append(pad_w - tile_size)
    if len(x_coords) == 0:
        x_coords = [0]
        
    tiles_info = []
    t_count = 0
    for ty in y_coords:
        for tx in x_coords:
            tiles_info.append({"tile_id": t_count, "x": tx, "y": ty})
            t_count += 1
            
    tile_count = len(tiles_info)
    print(f"[TILES] Generated {tile_count} tiles (pad size: {pad_w}x{pad_h})")
    
    # Load Model
    model = YOLO(str(MODEL_PATH))
    
    # Storage for results
    diagnostic_rows = []
    test_results_detailed = {}
    
    # 4. Run 12 Tests
    for prep in PREPROCESSING_TYPES:
        # Prepare full-size preprocessed padded image
        if prep == "CLAHE":
            gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            proc_single = clahe.apply(gray)
            padded_input = np.zeros((pad_h, pad_w), dtype=np.uint8)
            padded_input[0:h_align, 0:w_align] = proc_single
            is_3ch = False
        elif prep == "GRAY":
            gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            padded_input = np.zeros((pad_h, pad_w), dtype=np.uint8)
            padded_input[0:h_align, 0:w_align] = gray
            is_3ch = False
        elif prep == "BGR":
            padded_input = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
            padded_input[0:h_align, 0:w_align] = aligned_img
            is_3ch = True
            
        for conf in CONF_LEVELS:
            test_name = f"{prep}_conf{int(conf*100):03d}"
            print(f"\n--- Running Test: {prep} @ Conf {conf:.2f} ---")
            
            # 4.1 Raw YOLO Tiled Inference
            raw_detections = []
            for info in tiles_info:
                tid, tx, ty = info["tile_id"], info["x"], info["y"]
                if is_3ch:
                    tile_crop = padded_input[ty:ty+tile_size, tx:tx+tile_size]
                    tile_3ch = tile_crop
                else:
                    tile_crop = padded_input[ty:ty+tile_size, tx:tx+tile_size]
                    tile_3ch = cv2.cvtColor(tile_crop, cv2.COLOR_GRAY2BGR)
                    
                results = model.predict(source=tile_3ch, imgsz=640, conf=conf, verbose=False)
                for r in results:
                    names = r.names
                    for box in r.boxes:
                        cls, conf_val = int(box.cls[0]), float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        gx1, gy1 = tx + xyxy[0], ty + xyxy[1]
                        gx2, gy2 = tx + xyxy[2], ty + xyxy[3]
                        raw_detections.append({
                            "tile_id": tid, "class_id": cls, "class_name": names[cls],
                            "confidence": conf_val, "global_bbox": [float(gx1), float(gy1), float(gx2), float(gy2)]
                        })
                        
            raw_yolo_count = len(raw_detections)
            
            # 4.2 NMS
            global_boxes = [d["global_bbox"] for d in raw_detections]
            global_scores = [d["confidence"] for d in raw_detections]
            global_classes = [d["class_id"] for d in raw_detections]
            keep_indices = apply_nms(global_boxes, global_scores, global_classes, iou_threshold=IOU_THRESHOLD)
            nms_detections = [raw_detections[idx] for idx in keep_indices]
            nms_count = len(nms_detections)
            duplicates_removed = raw_yolo_count - nms_count
            
            # 4.3 Text/Pad Filtering
            final_valid_defects, negated_defects = filter_defects_with_text_and_pads(
                nms_detections, text_boxes, regular_vias
            )
            negated_count = len(negated_defects)
            final_valid_count = len(final_valid_defects)
            
            # Format detections string
            det_str_list = [f"{d['class_name']}:{d['confidence']:.3f}:[{int(d['global_bbox'][0])},{int(d['global_bbox'][1])},{int(d['global_bbox'][2])},{int(d['global_bbox'][3])}]" for d in raw_detections]
            detections_summary = "; ".join(det_str_list) if det_str_list else "NONE"
            
            diagnostic_rows.append({
                "preprocessing": prep,
                "confidence_threshold": f"{conf:.2f}",
                "tile_count": tile_count,
                "raw_yolo_count": raw_yolo_count,
                "nms_count": nms_count,
                "negated_count": negated_count,
                "final_valid_count": final_valid_count,
                "detections": detections_summary
            })
            
            test_results_detailed[test_name] = {
                "prep": prep, "conf": conf, "raw": raw_detections, "nms": nms_detections,
                "valid": final_valid_defects, "negated": negated_defects
            }
            
            print(f"  Raw YOLO: {raw_yolo_count} | NMS: {nms_count} | Negated: {negated_count} | Valid: {final_valid_count}")
            
            vis_canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
            vis_canvas[0:h_align, 0:w_align] = aligned_img
            
            # Draw text boxes faintly for spatial context
            for tb in text_boxes:
                cv2.rectangle(vis_canvas, (tb[0], tb[1]), (tb[2], tb[3]), (255, 200, 0), 1)
                cv2.putText(vis_canvas, "TEXT", (tb[0], max(12, tb[1] - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)
                            
            # Draw RAW YOLO detections in RED BOX
            if raw_detections:
                for d in raw_detections:
                    gx1, gy1, gx2, gy2 = map(int, d["global_bbox"])
                    cv2.rectangle(vis_canvas, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                    label = f"{d['class_name']} {d['confidence']:.2f}"
                    cv2.putText(vis_canvas, label, (gx1, max(15, gy1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(vis_canvas, "NO RAW YOLO DETECTIONS", (pad_w // 4, pad_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
                            
            # Add header banner
            header = np.zeros((50, pad_w, 3), dtype=np.uint8)
            cv2.putText(header, f"DIAGNOSTIC: {prep} | Conf: {conf:.2f} | Raw: {raw_yolo_count} | NMS: {nms_count} | Valid: {final_valid_count}",
                        (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            full_vis = np.vstack([header, vis_canvas])
            
            img_filename = f"{prep}_conf{int(conf*100):03d}.jpg"
            out_img_path = DIAG_OUT_DIR / img_filename
            cv2.imwrite(str(out_img_path), full_vis)
            cv2.imwrite(str(ART_DIAG_DIR / img_filename), full_vis)

    # 5. Write CSV
    csv_path = BASE_OUT_DIR / "black_pcb_diagnostic.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "preprocessing", "confidence_threshold", "tile_count",
            "raw_yolo_count", "nms_count", "negated_count", "final_valid_count", "detections"
        ])
        writer.writeheader()
        for row in diagnostic_rows:
            writer.writerow(row)
            
    shutil.copy(csv_path, ARTIFACTS_DIR / "black_pcb_diagnostic.csv")
    print(f"\n[SAVED] Diagnostic CSV: {csv_path}")

    # 6. Analyze Results for Report
    # 1. Highest raw YOLO detections
    prep_raw_counts = {p: sum(r["raw_yolo_count"] for r in diagnostic_rows if r["preprocessing"] == p) for p in PREPROCESSING_TYPES}
    max_raw_prep = max(prep_raw_counts, key=prep_raw_counts.get)
    
    # 2. Highest confidence defect across all tests
    all_raw_dets = []
    for row_name, data in test_results_detailed.items():
        for d in data["raw"]:
            all_raw_dets.append((data["prep"], data["conf"], d))
            
    if all_raw_dets:
        highest_conf_tuple = max(all_raw_dets, key=lambda x: x[2]["confidence"])
        highest_conf_prep = highest_conf_tuple[0]
        highest_conf_val = highest_conf_tuple[2]["confidence"]
        highest_conf_class = highest_conf_tuple[2]["class_name"]
        highest_conf_bbox = highest_conf_tuple[2]["global_bbox"]
    else:
        highest_conf_prep = "None"
        highest_conf_val = 0.0
        highest_conf_class = "None"
        highest_conf_bbox = []

    # 3 & 4. Confidence thresholds appearance
    dets_under_045 = [d for d in all_raw_dets if d[1] < 0.45]
    dets_at_045 = [d for d in all_raw_dets if d[1] == 0.45]
    
    # 5 & 6. Removal by NMS and Text/Pad filter
    nms_removals = sum(r["raw_yolo_count"] - r["nms_count"] for r in diagnostic_rows)
    filter_removals = sum(r["negated_count"] for r in diagnostic_rows)
    
    # 7 & 8. Comparison
    bgr_raw_total = prep_raw_counts["BGR"]
    clahe_raw_total = prep_raw_counts["CLAHE"]
    gray_raw_total = prep_raw_counts["GRAY"]
    
    # Determine Root Cause
    # Options:
    # A. PREPROCESSING ISSUE
    # B. CONFIDENCE THRESHOLD ISSUE
    # C. POST-PROCESSING FILTER ISSUE
    # D. MODEL / DATASET DOMAIN GAP
    # E. INCONCLUSIVE — MORE DATA REQUIRED
    
    total_raw_all = sum(r["raw_yolo_count"] for r in diagnostic_rows)
    total_valid_all = sum(r["final_valid_count"] for r in diagnostic_rows)
    
    if total_raw_all == 0:
        root_cause = "D. MODEL / DATASET DOMAIN GAP"
        root_cause_explanation = "The YOLOv8 model produced 0 raw detections across all 12 tests (even at 0.10 confidence in BGR, Grayscale, and CLAHE), demonstrating that the DeepPCB-trained model cannot identify features on this physical Black PCB without fine-tuning."
    elif bgr_raw_total > clahe_raw_total and clahe_raw_total == 0:
        root_cause = "A. PREPROCESSING ISSUE"
        root_cause_explanation = "Raw detections appeared in BGR/Grayscale but were eliminated by CLAHE preprocessing."
    elif len(dets_under_045) > 0 and len(dets_at_045) == 0:
        root_cause = "B. CONFIDENCE THRESHOLD ISSUE"
        root_cause_explanation = "Raw detections appeared consistently at lower confidence levels (0.10 - 0.30) but were cut off by the default 0.45 threshold."
    elif total_raw_all > 0 and total_valid_all == 0 and filter_removals > 0:
        root_cause = "C. POST-PROCESSING FILTER ISSUE"
        root_cause_explanation = "YOLO successfully detected candidates, but all detections were negated by the text/pad filter."
    else:
        root_cause = "D. MODEL / DATASET DOMAIN GAP"
        root_cause_explanation = "YOLO raw detections are sparse or low confidence across real physical features of the assembled Black PCB due to the synthetic bare-board training domain gap."

    report_text = f"""================================================================================
BLACK PCB CONTROLLED DIAGNOSTIC REPORT
================================================================================
Test Target Image      : {INPUT_IMG_PATH.name}
Registered Resolution  : {w_align}x{h_align} px
Tiles Generated        : {tile_count} (Tile Size: 640x640, Overlap: 96px)
Model Tested           : YOLOv8n ({MODEL_PATH.name})
Total Tests Run        : 12 (3 Preprocessing Methods x 4 Confidence Levels)

--------------------------------------------------------------------------------
12-TEST QUANTITATIVE MATRIX
--------------------------------------------------------------------------------
{'-'*88}
| Preprocessing | Conf  | Raw YOLO | After NMS | Negated (Text/Pad) | Final Valid | Detections
{'-'*88}
"""
    for r in diagnostic_rows:
        report_text += f"| {r['preprocessing']:<13} | {r['confidence_threshold']:<5} | {r['raw_yolo_count']:<8} | {r['nms_count']:<9} | {r['negated_count']:<18} | {r['final_valid_count']:<11} | {r['detections'][:30]}\n"
    report_text += f"{'-'*88}\n\n"

    report_text += f"""--------------------------------------------------------------------------------
ANSWERS TO DIAGNOSTIC QUESTIONS
--------------------------------------------------------------------------------
1. Which preprocessing produces the highest number of raw YOLO detections?
   Answer: {max_raw_prep} (CLAHE: {clahe_raw_total}, GRAY: {gray_raw_total}, BGR: {bgr_raw_total} total raw detections across all thresholds)

2. Which preprocessing produces the highest-confidence defect?
   Answer: {highest_conf_prep} (Class: {highest_conf_class}, Conf: {highest_conf_val:.3f}, BBox: {highest_conf_bbox})

3. Does the suspected defect appear at confidence < 0.45?
   Answer: {'YES (' + str(len(dets_under_045)) + ' detections found below 0.45)' if dets_under_045 else 'NO (0 detections found below 0.45)'}

4. Does the suspected defect appear at confidence >= 0.45?
   Answer: {'YES (' + str(len(dets_at_045)) + ' detections found at 0.45)' if dets_at_045 else 'NO (0 detections found at 0.45)'}

5. Is the detection removed by NMS?
   Answer: {'YES (' + str(nms_removals) + ' duplicate detections suppressed by NMS)' if nms_removals > 0 else 'NO (0 detections removed by NMS)'}

6. Is the detection removed by text/pad filtering?
   Answer: {'YES (' + str(filter_removals) + ' detections negated by text/pad filter)' if filter_removals > 0 else 'NO (0 detections negated)'}

7. Does original BGR perform better than CLAHE?
   Answer: {'YES' if bgr_raw_total > clahe_raw_total else ('NO (CLAHE had ' + str(clahe_raw_total) + ' vs BGR ' + str(bgr_raw_total) + ')') if clahe_raw_total > bgr_raw_total else 'EQUAL (Both produced ' + str(bgr_raw_total) + ' raw detections)'}

8. Does simple grayscale perform better than CLAHE?
   Answer: {'YES' if gray_raw_total > clahe_raw_total else ('NO (CLAHE had ' + str(clahe_raw_total) + ' vs GRAY ' + str(gray_raw_total) + ')') if clahe_raw_total > gray_raw_total else 'EQUAL (Both produced ' + str(gray_raw_total) + ' raw detections)'}

--------------------------------------------------------------------------------
ROOT CAUSE DETERMINATION
--------------------------------------------------------------------------------
ROOT CAUSE: {root_cause}

Detailed Evaluation:
{root_cause_explanation}

================================================================================
"""
    report_file = BASE_OUT_DIR / "black_pcb_diagnostic_report.txt"
    report_file.write_text(report_text)
    shutil.copy(report_file, ARTIFACTS_DIR / "black_pcb_diagnostic_report.txt")
    print(f"[SAVED] Diagnostic Report: {report_file}")
    print("\n" + report_text)

if __name__ == "__main__":
    run_diagnostic()
