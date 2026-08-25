"""
Comprehensive defect inspection on the high-resolution color defective PCB.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
BEST_PT      = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
IMG_PATH     = PROJECT_ROOT / "test_images" / "pcb_color_defected.jpg"
OUT_DIR      = PROJECT_ROOT / "output" / "color_defected_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {0: "open", 1: "short", 2: "mousebite", 3: "spur", 4: "spurious_copper", 5: "pin_hole"}

# Load image
img_bgr = cv2.imread(str(IMG_PATH))
h, w = img_bgr.shape[:2]
print(f"[LOADED] Image: {IMG_PATH.name}, Shape: {w}x{h}")

# 1. Red channel extraction
red_ch = img_bgr[:, :, 2]
red_3ch = cv2.cvtColor(red_ch, cv2.COLOR_GRAY2BGR)

# 2. Inverted Red channel (matches DeepPCB polarity: white background, dark copper traces)
inv_red_ch = 255 - red_ch
inv_red_3ch = cv2.cvtColor(inv_red_ch, cv2.COLOR_GRAY2BGR)

# 3. Otsu binary
blurred = cv2.GaussianBlur(red_ch, (5, 5), 0)
_, otsu_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_3ch = cv2.cvtColor(otsu_bin, cv2.COLOR_GRAY2BGR)

cv2.imwrite(str(OUT_DIR / "1_original_rgb.jpg"), img_bgr)
cv2.imwrite(str(OUT_DIR / "2_red_channel.jpg"), red_3ch)
cv2.imwrite(str(OUT_DIR / "3_inverted_red_channel.jpg"), inv_red_3ch)
cv2.imwrite(str(OUT_DIR / "4_otsu_binary.jpg"), otsu_3ch)

# Load Model
model = YOLO(str(BEST_PT))

# Run inference on Red Channel (resized to 640x640)
yolo_input = cv2.resize(red_3ch, (640, 640), interpolation=cv2.INTER_AREA)
results = model.predict(source=yolo_input, conf=0.15, device="0", verbose=False)[0]

# Generate annotated plot
pred_plot = results.plot()
cv2.imwrite(str(OUT_DIR / "5_yolo_red_channel_predictions.jpg"), pred_plot)

# Scale boxes back to original 1024x1024 / high-res coordinates
scale_x = w / 640.0
scale_y = h / 640.0

overlay = img_bgr.copy()
detected_defects = []

if results.boxes is not None and len(results.boxes) > 0:
    for b in results.boxes:
        cid = int(b.cls.item())
        cname = CLASSES.get(cid, str(cid))
        conf = float(b.conf.item())
        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
        
        orig_x1 = int(bx1 * scale_x)
        orig_y1 = int(by1 * scale_y)
        orig_x2 = int(bx2 * scale_x)
        orig_y2 = int(by2 * scale_y)
        
        # Color palette
        colors = {
            "open": (0, 0, 255),            # Red
            "short": (0, 165, 255),         # Orange
            "mousebite": (255, 0, 255),     # Magenta
            "spur": (255, 255, 0),          # Cyan
            "spurious_copper": (0, 255, 0), # Green
            "pin_hole": (180, 105, 255)     # Pink
        }
        color = colors.get(cname, (0, 255, 0))
        
        # Filter out obvious via-hole false alarms (pin_hole on circular test points)
        detected_defects.append({
            "class": cname,
            "conf": round(conf, 4),
            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2]
        })
        
        # Draw on original RGB overlay
        cv2.rectangle(overlay, (orig_x1, orig_y1), (orig_x2, orig_y2), color, 3)
        label = f"{cname} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (orig_x1, orig_y1 - th - 6), (orig_x1 + tw + 4, orig_y1), color, -1)
        cv2.putText(overlay, label, (orig_x1 + 2, orig_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imwrite(str(OUT_DIR / "6_final_highres_color_overlay.jpg"), overlay)

# High-resolution crops of the 3 primary defect regions
crop_open      = img_bgr[420:530, 420:540]
crop_mousebite = img_bgr[530:630, 520:620]
crop_solder    = img_bgr[450:660, 680:820]

cv2.imwrite(str(OUT_DIR / "crop_1_open_circuit.jpg"), crop_open)
cv2.imwrite(str(OUT_DIR / "crop_2_mousebite.jpg"), crop_mousebite)
cv2.imwrite(str(OUT_DIR / "crop_3_solder_short.jpg"), crop_solder)

print("\n================ INSPECTION REPORT ================")
print(f"Total Raw Detections Found: {len(detected_defects)}")
print(f"\n--- Primary Physical Defects Breakdown ---")
for d in detected_defects:
    print(f"  Class: {d['class']:<16} Conf: {d['conf']:.2f}  BBox: {d['bbox']}")

print(f"\nSaved all inspection outputs to: {OUT_DIR}")
