"""
=============================================================================
Full 50-Epoch Training & Post-Training Phase 1 Validation Pipeline
=============================================================================
Performs:
  1. Pre-training verification of all 11 requested items.
  2. Full 50-epoch training on NVIDIA RTX 2050 (device=0, workers=0, batch=8, imgsz=640).
  3. Extraction of post-training metrics, loss curves, confusion matrix, best epoch.
  4. Immediate Phase 1 validation on DeepPCB (Val & Test splits, 6 defect classes).
=============================================================================
"""

import sys
import os
import time
import json
import csv
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Directories & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
BASE_MODEL    = PROJECT_ROOT / "yolov8n.pt"
DATASET_YAML  = PROJECT_ROOT / "dataset" / "pcb.yaml"
TRAIN_IMG_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
VAL_IMG_DIR   = PROJECT_ROOT / "dataset" / "images" / "val"
TEST_IMG_DIR  = PROJECT_ROOT / "dataset" / "images" / "test"
TRAIN_LBL_DIR = PROJECT_ROOT / "dataset" / "labels" / "train"
VAL_LBL_DIR   = PROJECT_ROOT / "dataset" / "labels" / "val"
TEST_LBL_DIR  = PROJECT_ROOT / "dataset" / "labels" / "test"
TRAIN_OUT_DIR = PROJECT_ROOT / "output" / "training"
RUN_NAME      = "pcb_defect_yolov8n"
RUN_DIR       = TRAIN_OUT_DIR / RUN_NAME
WEIGHTS_DIR   = RUN_DIR / "weights"
BEST_PT       = WEIGHTS_DIR / "best.pt"
LAST_PT       = WEIGHTS_DIR / "last.pt"
PHASE1_DIR    = PROJECT_ROOT / "output" / "phase1_diagnostic"

CLASSES = {0: "open", 1: "short", 2: "mousebite", 3: "spur", 4: "spurious_copper", 5: "pin_hole"}
CLASS_NAMES = [CLASSES[i] for i in range(6)]

def sep(title="", width=74, ch="="):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{ch*pad} {title} {ch*(width - pad - len(title) - 2)}")
    else:
        print(ch * width)

def read_label_file(lbl_path: Path):
    if not lbl_path.exists():
        return []
    rows = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return rows

def count_instances(lbl_dir: Path):
    counts = defaultdict(int)
    for f in lbl_dir.glob("*.txt"):
        for r in read_label_file(f):
            counts[r[0]] += 1
    return counts

def run_pre_check():
    import torch
    sep("PRE-TRAINING VERIFICATION (11 ITEMS)")
    
    n_train_imgs = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
    n_val_imgs   = len(list(VAL_IMG_DIR.glob("*.jpg")))
    n_test_imgs  = len(list(TEST_IMG_DIR.glob("*.jpg")))
    
    train_inst = count_instances(TRAIN_LBL_DIR)
    val_inst   = count_instances(VAL_LBL_DIR)
    test_inst  = count_instances(TEST_LBL_DIR)
    
    total_annotations = sum(train_inst.values()) + sum(val_inst.values()) + sum(test_inst.values())
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Detected"
    
    print(f"1. Exact training dataset path   : {TRAIN_IMG_DIR}")
    print(f"2. Exact validation dataset path : {VAL_IMG_DIR}")
    print(f"3. Dataset YAML                  : {DATASET_YAML}")
    print(f"4. Number of train images        : {n_train_imgs} (Expected: 800) -> {'OK' if n_train_imgs == 800 else 'MISMATCH'}")
    print(f"5. Number of validation images   : {n_val_imgs} (Expected: 200) -> {'OK' if n_val_imgs == 200 else 'MISMATCH'}")
    print(f"6. Number of annotations         : {total_annotations} (Expected: 10,013) -> {'OK' if total_annotations == 10013 else 'MISMATCH'}")
    print("7. Exact YOLO class mapping      :")
    for cid in range(6):
        print(f"   * {cid} = {CLASSES[cid]}")
    print(f"8. Image size                    : 640")
    print(f"9. Batch size                    : 8")
    print(f"10. Epochs                       : 50")
    print(f"11. GPU                          : {gpu_name}")
    
    all_ok = (n_train_imgs == 800 and n_val_imgs == 200 and total_annotations == 10013 and torch.cuda.is_available())
    if not all_ok:
        print("\n[ERROR] Pre-training verification failed!")
        sys.exit(1)
    print("\n[SUCCESS] All 11 verification checks PASSED.")

if __name__ == "__main__":
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import torch

    # Step 1: Pre-training check
    run_pre_check()

    # Step 2: Start 50-epoch training
    sep("STARTING 50-EPOCH TRAINING ON GPU (device=0)")
    start_time = time.time()
    
    model = YOLO(str(BASE_MODEL))
    train_results = model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,  # Windows safe
        project=str(TRAIN_OUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
    )
    
    end_time = time.time()
    total_training_duration = end_time - start_time
    duration_mins = total_training_duration / 60.0
    
    sep("TRAINING COMPLETED")
    print(f"Total training duration : {duration_mins:.2f} minutes ({total_training_duration:.1f} seconds)")
    print(f"Best model saved at     : {BEST_PT}")
    print(f"Last model saved at     : {LAST_PT}")
    
    # Step 3: Analyze results.csv
    results_csv = RUN_DIR / "results.csv"
    best_epoch = 1
    best_map50 = 0.0
    final_epoch = 0
    final_metrics = {}
    
    if results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in reader]
        
        final_epoch = len(rows)
        for r in rows:
            try:
                ep = int(r.get("epoch", 0))
                m50 = float(r.get("metrics/mAP50(B)", 0.0))
                if m50 > best_map50:
                    best_map50 = m50
                    best_epoch = ep
            except:
                pass
        if rows:
            final_metrics = rows[-1]

    print(f"Final epoch completed   : {final_epoch} / 50")
    print(f"Best epoch recorded     : Epoch {best_epoch} (mAP50: {best_map50:.4f})")
    print(f"results.csv path        : {results_csv}")
    
    # Check generated curve artifacts
    loss_curve_path = RUN_DIR / "results.png"
    confusion_matrix_path = RUN_DIR / "confusion_matrix.png"
    confusion_matrix_norm = RUN_DIR / "confusion_matrix_normalized.png"
    print(f"Training loss curves    : {loss_curve_path} ({'Found' if loss_curve_path.exists() else 'Not found'})")
    print(f"Confusion matrix        : {confusion_matrix_path} ({'Found' if confusion_matrix_path.exists() else 'Not found'})")
    
    # Step 4: Run Phase 1 validation using newly trained best.pt
    sep("RUNNING POST-TRAINING PHASE 1 VALIDATION (best.pt)")
    best_model = YOLO(str(BEST_PT))
    
    # Validation split
    print("\n--- Evaluating best.pt on Validation Split (200 images) ---")
    val_metrics = best_model.val(
        data=str(DATASET_YAML),
        split="val",
        device="0",
        conf=0.25,
        iou=0.5,
        workers=0,
        plots=True,
        project=str(PHASE1_DIR),
        name="post_train_val_run",
        exist_ok=True,
    )
    
    val_mp    = float(val_metrics.box.mp)
    val_mr    = float(val_metrics.box.mr)
    val_map50 = float(val_metrics.box.map50)
    val_map   = float(val_metrics.box.map)
    
    print(f"\n[Validation Split Results]")
    print(f"  Precision (mean) : {val_mp:.4f}")
    print(f"  Recall (mean)    : {val_mr:.4f}")
    print(f"  mAP50            : {val_map50:.4f}")
    print(f"  mAP50-95         : {val_map:.4f}")
    
    print("\n[Per-Class Validation Metrics]")
    print(f"  {'Class':<20} {'Precision':>12} {'Recall':>10} {'AP50':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}")
    if hasattr(val_metrics.box, 'ap_class_index') and val_metrics.box.ap_class_index is not None:
        for i, cid in enumerate(val_metrics.box.ap_class_index):
            cname  = CLASSES.get(int(cid), f"class_{cid}")
            p_val  = float(val_metrics.box.p[i])
            r_val  = float(val_metrics.box.r[i])
            ap_val = float(val_metrics.box.ap50[i])
            print(f"  {cname:<20} {p_val:>12.4f} {r_val:>10.4f} {ap_val:>10.4f}")

    # Test split
    print("\n--- Evaluating best.pt on Test Split (500 images) ---")
    test_metrics = best_model.val(
        data=str(DATASET_YAML),
        split="test",
        device="0",
        conf=0.25,
        iou=0.5,
        workers=0,
        plots=True,
        project=str(PHASE1_DIR),
        name="post_train_test_run",
        exist_ok=True,
    )
    
    test_mp    = float(test_metrics.box.mp)
    test_mr    = float(test_metrics.box.mr)
    test_map50 = float(test_metrics.box.map50)
    test_map   = float(test_metrics.box.map)
    
    print(f"\n[Test Split Results]")
    print(f"  Precision (mean) : {test_mp:.4f}")
    print(f"  Recall (mean)    : {test_mr:.4f}")
    print(f"  mAP50            : {test_map50:.4f}")
    print(f"  mAP50-95         : {test_map:.4f}")

    print("\n[Per-Class Test Metrics]")
    print(f"  {'Class':<20} {'Precision':>12} {'Recall':>10} {'AP50':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10}")
    if hasattr(test_metrics.box, 'ap_class_index') and test_metrics.box.ap_class_index is not None:
        for i, cid in enumerate(test_metrics.box.ap_class_index):
            cname  = CLASSES.get(int(cid), f"class_{cid}")
            p_val  = float(test_metrics.box.p[i])
            r_val  = float(test_metrics.box.r[i])
            ap_val = float(test_metrics.box.ap50[i])
            print(f"  {cname:<20} {p_val:>12.4f} {r_val:>10.4f} {ap_val:>10.4f}")

    # Per-image representative defect test (3 images per class)
    sep("TEST 1 — Per-Image Defect Detection on DeepPCB Test Set (3 per class)")
    
    from phase1_diagnostic import pick_representative_images, yolo_norm_to_xyxy, draw_boxes_on_image
    test_picks = pick_representative_images(TEST_LBL_DIR, TEST_IMG_DIR, n_per_class=3)
    
    class_det_summary = defaultdict(lambda: {"images": 0, "detected": 0, "missed": 0})
    for cid in range(6):
        cname = CLASSES[cid]
        imgs = test_picks.get(cid, [])
        print(f"\n  --- Class {cid}: {cname} ({len(imgs)} test images) ---")
        for img_path in imgs:
            lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
            gt_rows = read_label_file(lbl_path)
            gt_classes = [CLASSES.get(r[0], f"cls_{r[0]}") for r in gt_rows]
            
            res = best_model.predict(source=str(img_path), conf=0.25, device="0", verbose=False)[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    det_cls = int(b.cls.item())
                    det_conf = float(b.conf.item())
                    det_xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
                    dets.append({
                        "class_name": CLASSES.get(det_cls, f"cls_{det_cls}"),
                        "class_id": det_cls,
                        "confidence": det_conf,
                        "bbox": det_xyxy
                    })
            
            target_detected = any(d["class_id"] == cid for d in dets)
            class_det_summary[cid]["images"] += 1
            if target_detected:
                class_det_summary[cid]["detected"] += 1
            else:
                class_det_summary[cid]["missed"] += 1
                
            det_str = ", ".join([f"{d['class_name']}@{d['confidence']:.2f}" for d in dets]) if dets else "NONE"
            print(f"    image: {img_path.name}")
            print(f"    ground_truth: {', '.join(gt_classes)}")
            print(f"    YOLO_detected: {det_str}")
            print(f"    target_detected: {'YES' if target_detected else 'NO'}")
            
    # Final Summary Table
    sep("FINAL PHASE 1 DIAGNOSTIC SUMMARY")
    print("""
┌──────────────────────────────┬──────────────────────────────────────────┐
│ Test                         │ Result                                   │
├──────────────────────────────┼──────────────────────────────────────────┤""")
    for cid in range(6):
        cname = CLASSES[cid]
        tot = class_det_summary[cid]["images"]
        det = class_det_summary[cid]["detected"]
        rate = (det / tot * 100) if tot > 0 else 0
        lbl = f"DeepPCB {cname.capitalize()}"
        res = f"{det}/{tot} detected ({rate:.0f}%)"
        print(f"│ {lbl:<28} │ {res:<40} │")
        
    print(f"│ {'Val mAP50':<28} │ {val_map50:<40.4f} │")
    print(f"│ {'Test mAP50':<28} │ {test_map50:<40.4f} │")
    print(f"│ {'Val Precision':<28} │ {val_mp:<40.4f} │")
    print(f"│ {'Val Recall':<28} │ {val_mr:<40.4f} │")
    print(f"│ {'Training Duration':<28} │ {duration_mins:<37.2f} min │")
    print(f"│ {'Best Epoch':<28} │ Epoch {best_epoch:<34} │")
    print("└──────────────────────────────┴──────────────────────────────────────────┘")
    
    overall_det = sum(class_det_summary[i]["detected"] for i in range(6))
    overall_tot = sum(class_det_summary[i]["images"] for i in range(6))
    overall_rate = (overall_det / max(overall_tot, 1)) * 100
    
    sep("EVALUATION CONCLUSION")
    print(f"Overall Test Detection Rate : {overall_det}/{overall_tot} ({overall_rate:.0f}%)")
    print(f"Validation mAP50             : {val_map50:.4f}")
    print(f"Test mAP50                   : {test_map50:.4f}")
    
    if val_map50 >= 0.80 and test_map50 >= 0.80 and overall_rate >= 80:
        print("\n[CONCLUSION A: PASS]")
        print("  The newly trained YOLOv8n model reliably detects DeepPCB defects across all 6 classes.")
        print("  The model is fully functional on the original dataset domain.")
    else:
        print("\n[CONCLUSION B: AUDIT REQUIRED]")
        print("  The model did not meet the desired threshold on DeepPCB domain.")
        
    print("\nPhase 1 execution complete.")
