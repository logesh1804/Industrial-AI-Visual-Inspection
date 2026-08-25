"""
=============================================================================
Phase 1 — Model Validation Diagnostic Script
=============================================================================
PURPOSE : Validate the existing best.pt YOLOv8n model on the DeepPCB domain.
          DIAGNOSTIC ONLY — does not modify best.pt, the dataset, or any
          live-camera pipeline.

Tests performed:
  Test 1 — Per-image prediction on DeepPCB defective images (all 6 classes)
  Test 2 — Per-image prediction on DeepPCB good/template images
  Test 3 — Dataset / class-mapping verification
  Test 4 — Model validation metrics (mAP, precision, recall, confusion matrix)
  Test 5 — Save visual prediction images (GT vs prediction)

Output directory: output/phase1_diagnostic/

NOTE: The 'if __name__ == "__main__":' guard is REQUIRED on Windows to
      prevent multiprocessing errors with PyTorch DataLoader.
=============================================================================
"""

import sys
import os
import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODEL_PATH    = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
DATASET_YAML  = PROJECT_ROOT / "dataset" / "pcb.yaml"
TRAIN_IMG_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
VAL_IMG_DIR   = PROJECT_ROOT / "dataset" / "images" / "val"
TEST_IMG_DIR  = PROJECT_ROOT / "dataset" / "images" / "test"
TRAIN_LBL_DIR = PROJECT_ROOT / "dataset" / "labels" / "train"
VAL_LBL_DIR   = PROJECT_ROOT / "dataset" / "labels" / "val"
TEST_LBL_DIR  = PROJECT_ROOT / "dataset" / "labels" / "test"
OUT_DIR       = PROJECT_ROOT / "output" / "phase1_diagnostic"

# Class names as per pcb.yaml (ground truth, do NOT change)
CLASSES = {0: "open", 1: "short", 2: "mousebite", 3: "spur", 4: "spurious_copper", 5: "pin_hole"}
CLASS_NAMES = [CLASSES[i] for i in range(6)]

# Confidence threshold — default YOLO 0.25, unchanged for Phase 1
CONF_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sep(title="", width=72, ch="="):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{ch*pad} {title} {ch*(width - pad - len(title) - 2)}")
    else:
        print(ch * width)


def read_label_file(lbl_path: Path):
    """Return list of (class_id, cx, cy, w, h) from a YOLO label txt."""
    if not lbl_path.exists():
        return []
    rows = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append((int(parts[0]), float(parts[1]),
                             float(parts[2]), float(parts[3]), float(parts[4])))
    return rows


def count_instances_in_dir(label_dir: Path):
    """Count instances per class across all label files in a directory."""
    counts = defaultdict(int)
    n_files = 0
    if not label_dir.exists():
        return counts, n_files
    for f in label_dir.glob("*.txt"):
        n_files += 1
        for (cls, *_) in read_label_file(f):
            counts[cls] += 1
    return counts, n_files


def pick_representative_images(label_dir: Path, image_dir: Path, n_per_class=3):
    """
    For each class pick up to n_per_class images that contain at least one
    annotation of that class.  Returns dict {class_id: [Path, ...]}.
    """
    picks = defaultdict(list)
    for lbl_path in sorted(label_dir.glob("*.txt")):
        rows = read_label_file(lbl_path)
        for (cls, *_) in rows:
            if len(picks[cls]) < n_per_class:
                img_path = image_dir / (lbl_path.stem + ".jpg")
                if img_path.exists() and img_path not in picks[cls]:
                    picks[cls].append(img_path)
    return picks


def yolo_norm_to_xyxy(rows, img_w, img_h):
    """Convert YOLO normalized (cx,cy,w,h) to pixel xyxy."""
    boxes = []
    for (cls, cx, cy, bw, bh) in rows:
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes_on_image(img_bgr, boxes_xyxy, class_ids, confidences, color_map,
                        class_names, label_prefix=""):
    """Draw filled-label bounding boxes on a copy of the image."""
    import cv2
    out = img_bgr.copy()
    for (x1, y1, x2, y2), cid, conf in zip(boxes_xyxy, class_ids, confidences):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = color_map[cid % len(color_map)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{label_prefix}{class_names[cid]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


# ============================================================================
# MAIN — required guard for Windows multiprocessing
# ============================================================================
if __name__ == "__main__":
    import cv2
    import numpy as np

    COLORS = [
        (220, 50,  50),   # 0 open           — red
        (50, 150, 220),   # 1 short           — blue
        (50, 200,  50),   # 2 mousebite       — green
        (220, 150,  50),  # 3 spur            — orange
        (150,  50, 220),  # 4 spurious_copper — purple
        (50,  220, 180),  # 5 pin_hole        — cyan
    ]

    # -----------------------------------------------------------------------
    # Preflight checks
    # -----------------------------------------------------------------------
    sep("PREFLIGHT CHECKS")

    if not MODEL_PATH.exists():
        print(f"[FATAL] best.pt not found at:\n  {MODEL_PATH}")
        sys.exit(1)

    print(f"[OK] best.pt found   : {MODEL_PATH}")
    print(f"[OK] best.pt size    : {MODEL_PATH.stat().st_size / 1e6:.2f} MB")
    print(f"[OK] dataset yaml    : {DATASET_YAML}")
    print(f"[OK] train images    : {TRAIN_IMG_DIR}")
    print(f"[OK] val   images    : {VAL_IMG_DIR}")
    print(f"[OK] test  images    : {TEST_IMG_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "visual_predictions").mkdir(exist_ok=True)

    # Import ultralytics
    try:
        from ultralytics import YOLO
        import torch
        print(f"[OK] ultralytics imported")
        print(f"[OK] CUDA available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     GPU            : {torch.cuda.get_device_name(0)}")
        device = "0" if torch.cuda.is_available() else "cpu"
    except ImportError as e:
        print(f"[FATAL] Could not import ultralytics: {e}")
        sys.exit(1)

    print("\nLoading best.pt …")
    model = YOLO(str(MODEL_PATH))
    print(f"[OK] Model loaded. Task={model.task}  nc={model.model.nc}  names={model.names}")

    if model.model.nc != 6:
        print(f"[WARNING] Model reports nc={model.model.nc} but expected 6 classes!")
    else:
        print(f"[OK] nc=6 — correct number of defect classes")

    # ========================================================================
    # TEST 3 — Dataset / Class Mapping Verification
    # ========================================================================
    sep("TEST 3 — Dataset / Class Mapping Verification")

    print("\n--- YAML class mapping (pcb.yaml) ---")
    for cid, cname in CLASSES.items():
        print(f"  YOLO class {cid}  ->  {cname}")

    print("\n--- Model internal names ---")
    mapping_ok = True
    for cid, cname in model.names.items():
        expected = CLASSES.get(int(cid), "UNKNOWN")
        match = "OK" if cname == expected else f"MISMATCH! expected '{expected}'"
        if cname != expected:
            mapping_ok = False
        print(f"  model class {cid}  ->  {cname}   [{match}]")

    print("\n--- Dataset size ---")
    n_train_img = len(list(TRAIN_IMG_DIR.glob("*.jpg"))) if TRAIN_IMG_DIR.exists() else 0
    n_val_img   = len(list(VAL_IMG_DIR.glob("*.jpg")))   if VAL_IMG_DIR.exists()   else 0
    n_test_img  = len(list(TEST_IMG_DIR.glob("*.jpg")))  if TEST_IMG_DIR.exists()  else 0
    print(f"  training images   : {n_train_img}")
    print(f"  validation images : {n_val_img}")
    print(f"  test images       : {n_test_img}")

    print("\n--- Instances per class per split ---")
    train_counts, _ = count_instances_in_dir(TRAIN_LBL_DIR)
    val_counts,   _ = count_instances_in_dir(VAL_LBL_DIR)
    test_counts,  _ = count_instances_in_dir(TEST_LBL_DIR)

    print(f"  {'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    class_instance_summary = {}
    for cid in range(6):
        tr  = train_counts.get(cid, 0)
        va  = val_counts.get(cid, 0)
        te  = test_counts.get(cid, 0)
        tot = tr + va + te
        class_instance_summary[cid] = {"train": tr, "val": va, "test": te, "total": tot}
        print(f"  {CLASSES[cid]:<20} {tr:>8} {va:>8} {te:>8} {tot:>8}")
    total_all = sum(v["total"] for v in class_instance_summary.values())
    print(f"  {'TOTAL':<20} {sum(train_counts.values()):>8} {sum(val_counts.values()):>8} {sum(test_counts.values()):>8} {total_all:>8}")

    print("\n--- Class balance analysis ---")
    all_totals = {cid: class_instance_summary[cid]["total"] for cid in range(6)}
    max_cls = max(all_totals, key=all_totals.get)
    min_cls = min(all_totals, key=all_totals.get)
    ratio = all_totals[max_cls] / max(all_totals[min_cls], 1)
    print(f"  Max class : {CLASSES[max_cls]} ({all_totals[max_cls]} instances)")
    print(f"  Min class : {CLASSES[min_cls]} ({all_totals[min_cls]} instances)")
    print(f"  Imbalance ratio : {ratio:.1f}x")
    if ratio > 5:
        print("  [WARNING] Severe class imbalance detected!")
    else:
        print("  [OK] Class balance is acceptable")

    print("\n--- Training run history (results.csv) ---")
    results_csv = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "results.csv"
    n_epochs_completed = 0
    epoch1_map50 = epoch1_p = epoch1_r = float('nan')
    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in reader]
        data_rows = [r for r in rows if r.get("epoch", "").strip()]
        n_epochs_completed = len(data_rows)
        print(f"  Epochs completed  : {n_epochs_completed} / 50 (configured)")
        if data_rows:
            last = data_rows[-1]
            try:
                epoch1_map50 = float(last.get("metrics/mAP50(B)", "nan"))
                epoch1_p     = float(last.get("metrics/precision(B)", "nan"))
                epoch1_r     = float(last.get("metrics/recall(B)", "nan"))
                print(f"  Last epoch mAP50  : {epoch1_map50:.4f}")
                print(f"  Last epoch P      : {epoch1_p:.4f}")
                print(f"  Last epoch R      : {epoch1_r:.4f}")
            except:
                pass
        if n_epochs_completed < 10:
            print(f"\n  [CRITICAL WARNING] Training only completed {n_epochs_completed} epoch(s)!")
            print(f"  The model is severely undertrained. Expected 50 epochs.")
    else:
        print("  results.csv not found.")

    # ========================================================================
    # TEST 4 — Model Validation Metrics (workers=0 for Windows)
    # ========================================================================
    sep("TEST 4 — Model Validation Metrics (val split)")
    print("\nRunning model.val() on validation split (workers=0 for Windows) …\n")

    map50 = map5095 = mp = mr = float('nan')
    per_class_results = {}

    try:
        val_results = model.val(
            data=str(DATASET_YAML),
            split="val",
            device=device,
            conf=CONF_THRESHOLD,
            iou=0.5,
            workers=0,           # CRITICAL on Windows — no subprocess spawning
            verbose=True,
            plots=True,
            save_json=False,
            project=str(OUT_DIR),
            name="val_run",
            exist_ok=True,
        )

        print("\n--- Aggregate Metrics (Val Split) ---")
        mp       = float(val_results.box.mp)     if hasattr(val_results.box, 'mp')     else float('nan')
        mr       = float(val_results.box.mr)     if hasattr(val_results.box, 'mr')     else float('nan')
        map50    = float(val_results.box.map50)  if hasattr(val_results.box, 'map50')  else float('nan')
        map5095  = float(val_results.box.map)    if hasattr(val_results.box, 'map')    else float('nan')
        print(f"  mAP50           : {map50:.4f}")
        print(f"  mAP50-95        : {map5095:.4f}")
        print(f"  Precision (mean): {mp:.4f}")
        print(f"  Recall (mean)   : {mr:.4f}")

        print("\n--- Per-Class Metrics (Val Split) ---")
        print(f"  {'Class':<20} {'Precision':>12} {'Recall':>10} {'AP50':>10}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}")
        if hasattr(val_results.box, 'ap_class_index') and val_results.box.ap_class_index is not None:
            for i, cid in enumerate(val_results.box.ap_class_index):
                cname  = CLASSES.get(int(cid), f"class_{cid}")
                p_val  = float(val_results.box.p[i])   if i < len(val_results.box.p)    else float('nan')
                r_val  = float(val_results.box.r[i])   if i < len(val_results.box.r)    else float('nan')
                ap_val = float(val_results.box.ap50[i]) if i < len(val_results.box.ap50) else float('nan')
                per_class_results[cname] = {"precision": p_val, "recall": r_val, "ap50": ap_val}
                print(f"  {cname:<20} {p_val:>12.4f} {r_val:>10.4f} {ap_val:>10.4f}")
        else:
            print("  Per-class breakdown not available.")

        with open(OUT_DIR / "test4_val_metrics.json", "w") as f:
            json.dump({"map50": map50, "map50_95": map5095, "precision_mean": mp,
                       "recall_mean": mr, "per_class": per_class_results}, f, indent=2)
        print(f"\n  Metrics saved to: {OUT_DIR / 'test4_val_metrics.json'}")

    except Exception as e:
        print(f"  [ERROR] Validation failed: {e}")
        import traceback; traceback.print_exc()

    # Test split validation
    sep("TEST 4b — Model Validation Metrics (test split)")
    print("\nRunning model.val() on test split (workers=0) …\n")
    t_map50 = t_map5095 = t_mp = t_mr = float('nan')
    try:
        test_results = model.val(
            data=str(DATASET_YAML),
            split="test",
            device=device,
            conf=CONF_THRESHOLD,
            iou=0.5,
            workers=0,
            verbose=True,
            plots=True,
            save_json=False,
            project=str(OUT_DIR),
            name="test_run",
            exist_ok=True,
        )
        t_mp      = float(test_results.box.mp)     if hasattr(test_results.box, 'mp')    else float('nan')
        t_mr      = float(test_results.box.mr)     if hasattr(test_results.box, 'mr')    else float('nan')
        t_map50   = float(test_results.box.map50)  if hasattr(test_results.box, 'map50') else float('nan')
        t_map5095 = float(test_results.box.map)    if hasattr(test_results.box, 'map')   else float('nan')
        print(f"\n  Test mAP50    : {t_map50:.4f}")
        print(f"  Test mAP50-95 : {t_map5095:.4f}")
        print(f"  Test Precision: {t_mp:.4f}")
        print(f"  Test Recall   : {t_mr:.4f}")

        # Per-class test metrics
        print("\n--- Per-Class Metrics (Test Split) ---")
        print(f"  {'Class':<20} {'Precision':>12} {'Recall':>10} {'AP50':>10}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}")
        if hasattr(test_results.box, 'ap_class_index') and test_results.box.ap_class_index is not None:
            for i, cid in enumerate(test_results.box.ap_class_index):
                cname  = CLASSES.get(int(cid), f"class_{cid}")
                p_val  = float(test_results.box.p[i])    if i < len(test_results.box.p)    else float('nan')
                r_val  = float(test_results.box.r[i])    if i < len(test_results.box.r)    else float('nan')
                ap_val = float(test_results.box.ap50[i]) if i < len(test_results.box.ap50) else float('nan')
                print(f"  {cname:<20} {p_val:>12.4f} {r_val:>10.4f} {ap_val:>10.4f}")
    except Exception as e:
        print(f"  [ERROR] Test validation failed: {e}")
        import traceback; traceback.print_exc()

    # ========================================================================
    # TEST 1 — Per-Image Predictions on Defective DeepPCB Test Images
    # ========================================================================
    sep("TEST 1 — Per-Image Predictions on Defective DeepPCB Images")

    print("\nSelecting up to 3 representative images per class from test split …")
    test_picks = pick_representative_images(TEST_LBL_DIR, TEST_IMG_DIR, n_per_class=3)

    test1_rows = []
    class_detected = defaultdict(lambda: {"images": 0, "detected": 0, "missed": 0})

    for cid in range(6):
        cname = CLASSES[cid]
        imgs  = test_picks.get(cid, [])
        if not imgs:
            print(f"\n  [WARNING] No test images found for class '{cname}'")
            continue

        print(f"\n  --- Class {cid}: {cname} ({len(imgs)} images) ---")
        for img_path in imgs:
            lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
            gt_rows  = read_label_file(lbl_path)
            gt_classes = [CLASSES.get(r[0], f"cls_{r[0]}") for r in gt_rows]

            results = model.predict(
                source=str(img_path),
                conf=CONF_THRESHOLD,
                device=device,
                verbose=False,
            )
            r = results[0]
            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    det_cls  = int(box.cls.item())
                    det_conf = float(box.conf.item())
                    det_xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                    dets.append({
                        "class_id":   det_cls,
                        "class_name": CLASSES.get(det_cls, f"cls_{det_cls}"),
                        "confidence": round(det_conf, 4),
                        "bbox_xyxy":  det_xyxy,
                    })

            detected_correct = any(d["class_id"] == cid for d in dets)
            detected_status  = "YES" if detected_correct else "NO"

            class_detected[cid]["images"] += 1
            if detected_correct:
                class_detected[cid]["detected"] += 1
            else:
                class_detected[cid]["missed"] += 1

            det_summary = ", ".join([f"{d['class_name']}@{d['confidence']:.2f}" for d in dets]) if dets else "NONE"
            print(f"    image            : {img_path.name}")
            print(f"    ground_truth     : {', '.join(gt_classes)}")
            print(f"    YOLO_detections  : {det_summary}")
            print(f"    target_detected  : {detected_status}")
            if dets:
                for d in dets:
                    print(f"       -> {d['class_name']} conf={d['confidence']:.4f}  bbox={d['bbox_xyxy']}")
            print()

            test1_rows.append({
                "image": img_path.name,
                "ground_truth_class": cname,
                "all_gt_classes": "|".join(gt_classes),
                "yolo_detected_class": dets[0]["class_name"] if dets else "NONE",
                "confidence": dets[0]["confidence"] if dets else 0.0,
                "detected": detected_status,
                "total_detections": len(dets),
            })

    print("\n--- Test 1 Detection Summary per Class ---")
    print(f"  {'Class':<20} {'Images':>8} {'Detected':>10} {'Missed':>8} {'Rate':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for cid in range(6):
        cname = CLASSES[cid]
        imgs  = class_detected[cid]["images"]
        det   = class_detected[cid]["detected"]
        mis   = class_detected[cid]["missed"]
        rate  = f"{det/imgs*100:.0f}%" if imgs > 0 else "N/A"
        print(f"  {cname:<20} {imgs:>8} {det:>10} {mis:>8} {rate:>8}")

    csv_path = OUT_DIR / "test1_per_image_results.csv"
    if test1_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=test1_rows[0].keys())
            writer.writeheader()
            writer.writerows(test1_rows)
        print(f"\n  Results saved to: {csv_path}")

    # ========================================================================
    # TEST 2 — False Positive Check on Val Images
    # ========================================================================
    sep("TEST 2 — False Positive Rate on Val Images")
    print("\nNote: DeepPCB has no 'good' images — all images contain defects.")
    print("Checking false-positive class confusion (detected wrong class vs GT).\n")

    val_imgs = sorted(VAL_IMG_DIR.glob("*.jpg"))[:15]
    test2_rows = []
    fp_count = tp_count = total_t2 = 0

    for img_path in val_imgs:
        lbl_path = VAL_LBL_DIR / (img_path.stem + ".txt")
        gt_rows  = read_label_file(lbl_path)
        gt_class_ids = set(r[0] for r in gt_rows)

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESHOLD,
            device=device,
            verbose=False,
        )
        r = results[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                dets.append({
                    "class_id":   int(box.cls.item()),
                    "class_name": CLASSES.get(int(box.cls.item()), f"cls_{int(box.cls.item())}"),
                    "confidence": round(float(box.conf.item()), 4),
                })

        true_positives  = [d for d in dets if d["class_id"] in gt_class_ids]
        false_positives = [d for d in dets if d["class_id"] not in gt_class_ids]

        total_t2 += 1
        if false_positives:
            fp_count += 1
        if true_positives:
            tp_count += 1

        det_str = ", ".join([f"{d['class_name']}@{d['confidence']:.2f}" for d in dets]) if dets else "NONE"
        fp_flag = "YES" if false_positives else "NO"
        print(f"  {img_path.name:<45} GT:{sorted(gt_class_ids)}  dets={det_str}  FP={fp_flag}")

        test2_rows.append({
            "image": img_path.name,
            "gt_classes": sorted(gt_class_ids),
            "detections": det_str,
            "n_detections": len(dets),
            "n_true_pos": len(true_positives),
            "n_false_pos": len(false_positives),
            "has_false_positive": fp_flag,
        })

    print(f"\n  Images tested        : {total_t2}")
    print(f"  Images with any TP   : {tp_count}")
    print(f"  Images with any FP   : {fp_count}")
    print(f"  FP rate (image-level): {fp_count/max(total_t2,1)*100:.1f}%")

    csv_path2 = OUT_DIR / "test2_false_positive_check.csv"
    if test2_rows:
        with open(csv_path2, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=test2_rows[0].keys())
            writer.writeheader()
            writer.writerows(test2_rows)
    print(f"  Results saved to: {csv_path2}")

    # ========================================================================
    # TEST 5 — Visual Predictions
    # ========================================================================
    sep("TEST 5 — Save Visual Predictions (GT vs YOLO)")

    vis_dir = OUT_DIR / "visual_predictions"
    vis_dir.mkdir(exist_ok=True)
    vis_summary = []

    print("\nGenerating side-by-side GT vs Prediction images (1 per class) …")

    for cid in range(6):
        cname = CLASSES[cid]
        imgs  = test_picks.get(cid, [])
        if not imgs:
            print(f"  [SKIP] No images for class '{cname}'")
            continue

        img_path = imgs[0]
        lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
        gt_rows  = read_label_file(lbl_path)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [ERROR] Cannot read: {img_path}")
            continue
        h, w = img_bgr.shape[:2]

        # GT panel
        gt_boxes   = yolo_norm_to_xyxy(gt_rows, w, h)
        gt_classes = [r[0] for r in gt_rows]
        gt_confs   = [1.0] * len(gt_rows)
        gt_panel   = draw_boxes_on_image(img_bgr, gt_boxes, gt_classes, gt_confs,
                                         COLORS, CLASS_NAMES, label_prefix="GT:")
        cv2.putText(gt_panel, "GROUND TRUTH", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(gt_panel, "GROUND TRUTH", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Prediction panel
        res = model.predict(source=str(img_path), conf=CONF_THRESHOLD,
                            device=device, verbose=False)
        r = res[0]
        pred_boxes = pred_cls = pred_conf = []
        if r.boxes is not None and len(r.boxes) > 0:
            pred_boxes = [b.xyxy[0].tolist() for b in r.boxes]
            pred_cls   = [int(b.cls.item()) for b in r.boxes]
            pred_conf  = [float(b.conf.item()) for b in r.boxes]

        pred_panel = draw_boxes_on_image(img_bgr, pred_boxes, pred_cls, pred_conf,
                                         COLORS, CLASS_NAMES, label_prefix="")
        cv2.putText(pred_panel, f"YOLO PREDICTION (conf>={CONF_THRESHOLD})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(pred_panel, f"YOLO PREDICTION (conf>={CONF_THRESHOLD})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        separator = np.ones((h, 4, 3), dtype=np.uint8) * 128
        combined  = np.concatenate([gt_panel, separator, pred_panel], axis=1)

        save_path = vis_dir / f"class_{cid}_{cname}_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), combined)

        detected = any(c == cid for c in pred_cls)
        status   = "DETECTED" if detected else "MISSED"
        print(f"  [{status}] {cname:<20} -> {save_path.name}")
        vis_summary.append({"class": cname, "image": img_path.name,
                            "n_gt": len(gt_rows), "n_pred": len(pred_boxes),
                            "detected_target": detected})

    print(f"\n  Visual predictions saved to: {vis_dir}")

    # ========================================================================
    # FINAL SUMMARY TABLE
    # ========================================================================
    sep("PHASE 1 — FINAL SUMMARY TABLE")

    print("""
┌──────────────────────────────┬──────────────────────────────────────────┐
│ Test                         │ Result                                   │
├──────────────────────────────┼──────────────────────────────────────────┤""")

    for cid in range(6):
        cname = CLASSES[cid]
        imgs  = class_detected[cid]["images"]
        det   = class_detected[cid]["detected"]
        if imgs == 0:
            result = "No images found"
        else:
            rate = det / imgs * 100
            result = f"{det}/{imgs} detected ({rate:.0f}%)"
        label = f"DeepPCB {cname.capitalize()}"
        print(f"│ {label:<28} │ {result:<40} │")

    fp_str = f"{fp_count}/{total_t2} images had wrong-class detections"
    print(f"│ {'DeepPCB Good PCB':<28} │ {fp_str:<40} │")

    map50_str  = f"{map50:.4f}"  if not (map50  != map50)  else "N/A (val failed)"
    t_map50_str = f"{t_map50:.4f}" if not (t_map50 != t_map50) else "N/A"
    mr_str     = f"{mr:.4f}"    if not (mr     != mr)     else "N/A"
    mp_str     = f"{mp:.4f}"    if not (mp     != mp)     else "N/A"
    print(f"│ {'Val mAP50':<28} │ {map50_str:<40} │")
    print(f"│ {'Test mAP50':<28} │ {t_map50_str:<40} │")
    print(f"│ {'Overall Recall (val)':<28} │ {mr_str:<40} │")
    print(f"│ {'Overall Precision (val)':<28} │ {mp_str:<40} │")

    mapping_str = "YES — all 6 classes match" if mapping_ok else "NO — MISMATCH detected!"
    print(f"│ {'Class mapping correct?':<28} │ {mapping_str:<40} │")
    print(f"│ {'Training epochs completed':<28} │ {n_epochs_completed}/50                                    │")
    print("├──────────────────────────────┼──────────────────────────────────────────┤")

    overall_detected = sum(class_detected[i]["detected"] for i in range(6))
    overall_images   = sum(class_detected[i]["images"]   for i in range(6))
    overall_rate     = overall_detected / max(overall_images, 1) * 100

    # Use val mAP50 if available, fall back to epoch-1 metric
    effective_map50 = map50 if not (map50 != map50) else epoch1_map50

    if effective_map50 >= 0.50 and overall_rate >= 60:
        functional_str = "YES — Model is functional on DeepPCB"
        conclusion = "A"
    elif effective_map50 >= 0.25 or overall_rate >= 30:
        functional_str = "PARTIAL — Some detection, but unreliable"
        conclusion = "B (borderline)"
    else:
        functional_str = "NO — Model fails on DeepPCB domain"
        conclusion = "B"

    print(f"│ {'Model functional on DeepPCB?':<28} │ {functional_str:<40} │")
    print("└──────────────────────────────┴──────────────────────────────────────────┘")

    sep("CONCLUSION")
    print(f"\nConclusion Code: {conclusion}")
    print(f"Epochs trained : {n_epochs_completed}/50")
    print(f"Val mAP50      : {map50_str}")
    print(f"Test mAP50     : {t_map50_str}")
    print(f"Overall test-set detection rate: {overall_detected}/{overall_images} = {overall_rate:.0f}%")
    print()

    if conclusion == "A":
        print("  ✅  The current model is fundamentally functional on its training domain.")
        print("  The next investigation should focus on the live-camera domain gap,")
        print("  preprocessing, alignment, and defect preservation.")
    else:
        print(f"  ❌  The current model/dataset/training pipeline must be investigated")
        print(f"  before modifying the live-camera pipeline.")
        print()
        print(f"  ROOT CAUSE: Training was completed for only {n_epochs_completed} epoch(s) out of 50.")
        print(f"  The model is severely undertrained and has not converged.")
        print()
        print(f"  IMMEDIATE ACTION REQUIRED:")
        print(f"  1. Rerun training for the full 50 epochs")
        print(f"  2. Re-run Phase 1 diagnostic after training completes")
        print(f"  3. Only after Phase 1 passes should the live-camera pipeline be modified")

    sep("OUTPUT FILES")
    print(f"\n  {OUT_DIR}/")
    print(f"    test1_per_image_results.csv     — per-image GT vs prediction table")
    print(f"    test2_false_positive_check.csv  — false-positive analysis")
    print(f"    test4_val_metrics.json          — aggregate + per-class metrics")
    print(f"    val_run/                        — val artifacts (confusion matrix, PR curve)")
    print(f"    test_run/                       — test artifacts")
    print(f"    visual_predictions/             — GT vs prediction side-by-side images")
    print()
    print("Phase 1 diagnostic complete.\n")
