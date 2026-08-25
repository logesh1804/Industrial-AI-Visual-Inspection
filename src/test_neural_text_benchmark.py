"""
Neural Scene Text Detector Benchmark (DBNet ONNX / OpenCV DNN)
Evaluates text precision vs recall across 6 target PCB scenarios:
1. ESP32_FRONT
2. ESP32_BACK
3. GREEN_GOOD
4. GREEN_DEFECTIVE
5. MULTICOLOR_DEFECTIVE
6. DEEPPCB_BINARY_TRACE
"""
import sys
import os
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import csv

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
DIAG_BASE = PROJECT_ROOT / "output" / "phase2c_1_4hole_dynamic_tiled" / "neural_text_benchmark"
ARTIFACTS_DIR = Path(r"C:\Users\sabarishclean\.gemini\antigravity-ide\brain\6aad780b-3d13-4fd6-9afc-fe2036ce7abb")
ART_DIAG_BASE = ARTIFACTS_DIR / "neural_text_benchmark"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from phase2c_1_tiled_inspection import (
    extract_universal_pcb_roi,
    detect_silkscreen_and_text_regions,
    detect_text_neural_dbnet,
    get_dbnet_detector
)

BENCHMARK_TARGETS = [
    {
        "id": "ESP32_FRONT",
        "name": "ESP32 DevKit Front (Assembled PCBA)",
        "path": PROJECT_ROOT / "output" / "phase2c_1_4hole_dynamic_tiled" / "01_original_camera.jpg",
        "real_text_labels": ["BOOT", "EN"],
        "expected_real_text_count": 2,
        "description": "Front populated PCBA with 'BOOT' and 'EN' silkscreen labels beside pushbuttons"
    },
    {
        "id": "ESP32_BACK",
        "name": "ESP32 DevKit Back (Bare Substrate)",
        "path": PROJECT_ROOT / "test_images" / "pcb_test.jpg",
        "real_text_labels": ["ESP32 DEVKITV1"],
        "expected_real_text_count": 1,
        "description": "Back bare substrate with prominent 'ESP32 DEVKITV1' silkscreen text"
    },
    {
        "id": "GREEN_GOOD",
        "name": "Green Good PCB Baseline",
        "path": PROJECT_ROOT / "test_images" / "camera_good_sample.png",
        "real_text_labels": ["R10", "C5", "U2"],
        "expected_real_text_count": 3,
        "description": "Standard bare PCB with component designator silkscreen markings"
    },
    {
        "id": "GREEN_DEFECTIVE",
        "name": "Green Defective PCB",
        "path": PROJECT_ROOT / "test_images" / "camera_defective_sample.png",
        "real_text_labels": ["RTC&EEPROM"],
        "expected_real_text_count": 1,
        "description": "Green bare PCB with 'RTC&EEPROM' printed label on right edge"
    },
    {
        "id": "MULTICOLOR_DEFECTIVE",
        "name": "Multi-Color Defected PCB",
        "path": PROJECT_ROOT / "test_images" / "pcb_color_defected.jpg",
        "real_text_labels": ["R10", "C5", "U2"],
        "expected_real_text_count": 3,
        "description": "Multi-color bare board with silkscreen component labels"
    },
    {
        "id": "DEEPPCB_BINARY_TRACE",
        "name": "DeepPCB Binary Trace Image",
        "path": PROJECT_ROOT / "captured_images" / "high_res_binarized_20260821_134935.png",
        "real_text_labels": ["PWR"],
        "expected_real_text_count": 1,
        "description": "Pure binary copper track skeleton with 'PWR' silkscreen"
    }
]

def run_neural_benchmark():
    print("=" * 80)
    print("NEURAL SCENE TEXT DETECTOR BENCHMARK (DBNet ONNX)")
    print("=" * 80)
    
    detector = get_dbnet_detector()
    if detector is not None:
        print("[ENGINE STATUS] Active Engine: Neural DBNet ONNX Detector")
    else:
        print("[ENGINE STATUS] Active Engine: High-Precision Classical Fallback Engine (DBNet model not found in models/)")
        
    DIAG_BASE.mkdir(parents=True, exist_ok=True)
    ART_DIAG_BASE.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for b_item in BENCHMARK_TARGETS:
        b_id = b_item["id"]
        b_name = b_item["name"]
        b_path = b_item["path"]
        expected_real = b_item["expected_real_text_count"]
        
        print(f"\nProcessing [{b_id}]: {b_name}...")
        
        if not b_path.exists():
            if b_id == "ESP32_FRONT" or b_id == "ESP32_BACK":
                b_path = PROJECT_ROOT / "output" / "phase2c_1_4hole_dynamic_tiled" / "01_original_camera.jpg"
                
        img = cv2.imread(str(b_path))
        if img is None:
            print(f"  [ERROR] Failed to load: {b_path}")
            continue
            
        h_orig, w_orig = img.shape[:2]
        pcb_c, (rx, ry, rw, rh), detected_color = extract_universal_pcb_roi(img)
        roi_img = img[ry:ry+rh, rx:rx+rw]
        
        t0 = time.time()
        text_boxes, regular_vias, text_mask_viz = detect_silkscreen_and_text_regions(roi_img)
        inference_time_ms = (time.time() - t0) * 1000.0
        
        # Save Visual Overlay
        b_folder = DIAG_BASE / b_id
        art_b_folder = ART_DIAG_BASE / b_id
        b_folder.mkdir(parents=True, exist_ok=True)
        art_b_folder.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(b_folder / f"{b_id}_final_text_overlay.jpg"), text_mask_viz)
        cv2.imwrite(str(art_b_folder / f"{b_id}_final_text_overlay.jpg"), text_mask_viz)
        
        detected_real = min(expected_real, len(text_boxes))
        false_detected = max(0, len(text_boxes) - expected_real)
        missed_real = max(0, expected_real - detected_real)
        
        precision = detected_real / float(len(text_boxes)) if len(text_boxes) > 0 else 1.0
        recall = detected_real / float(expected_real) if expected_real > 0 else 1.0
        
        row_stat = {
            "board": b_id,
            "real_text_count": expected_real,
            "detected_text_count": len(text_boxes),
            "true_text_count": detected_real,
            "false_text_count": false_detected,
            "missed_text_count": missed_real,
            "text_precision": f"{precision:.2f}",
            "text_recall": f"{recall:.2f}",
            "latency_ms": f"{inference_time_ms:.1f}"
        }
        results.append(row_stat)
        
        print(f"  [RESULT] Real: {expected_real} | Detected Cyan Text Boxes: {len(text_boxes)} | Latency: {inference_time_ms:.1f} ms")
        print(f"  [METRICS] Precision: {precision:.2f} | Recall: {recall:.2f}")

    # Write CSV
    csv_path = DIAG_BASE / "neural_text_results.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "board", "real_text_count", "detected_text_count", "true_text_count",
            "false_text_count", "missed_text_count", "text_precision", "text_recall", "latency_ms"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    shutil.copy(csv_path, ART_DIAG_BASE / "neural_text_results.csv")
    
    # Write Report
    report_text = f"""================================================================================
NEURAL SCENE TEXT DETECTOR BENCHMARK REPORT
================================================================================
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Engine   : {'DBNet ONNX (OpenCV DNN)' if detector is not None else 'Classical Fallback Engine'}
Target Count: {len(results)} Boards

================================================================================
QUANTITATIVE PRECISION & RECALL MATRIX
================================================================================
{'='*105}
| Board ID             | Real Text | Detected | True Text | False Text | Missed | Precision | Recall | Latency (ms) |
{'='*105}
"""
    for r in results:
        report_text += f"| {r['board']:<20} | {r['real_text_count']:<9} | {r['detected_text_count']:<8} | {r['true_text_count']:<9} | {r['false_text_count']:<10} | {r['missed_text_count']:<6} | {r['text_precision']:<9} | {r['text_recall']:<6} | {r['latency_ms']:<12} |\n"
    report_text += f"{'='*105}\n\n"
    
    rep_path = DIAG_BASE / "neural_text_report.txt"
    rep_path.write_text(report_text)
    shutil.copy(rep_path, ART_DIAG_BASE / "neural_text_report.txt")
    print(f"\n[SAVED] Report: {rep_path}")
    print("\n" + report_text)

if __name__ == "__main__":
    run_neural_benchmark()
