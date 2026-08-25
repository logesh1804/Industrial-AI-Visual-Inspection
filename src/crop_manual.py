import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = PROJECT_ROOT / "captured_images"
OUTPUT_DIR = PROJECT_ROOT / "output" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CALIBRATION: Adjust these parameters to crop your PCB
# ==========================================
# Coordinates are in pixels: [y_start, y_end, x_start, x_end]
CROP_BOX = {
    "y_start": 150,
    "y_end": 400,
    "x_start": 200,
    "x_end": 550
}

def main():
    # Load test image
    img_path = CAPTURE_DIR / "pcb_20260806_120700.png"
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
        
    img = cv2.imread(str(img_path))
    h, w, c = img.shape
    print(f"Loaded image size: {w}x{h}")
    
    # 1. Apply the crop
    ys, ye = CROP_BOX["y_start"], CROP_BOX["y_end"]
    xs, xe = CROP_BOX["x_start"], CROP_BOX["x_end"]
    
    # Ensure crop coordinates are within image bounds
    ys, ye = max(0, ys), min(h, ye)
    xs, xe = max(0, xs), min(w, xe)
    
    cropped = img[ys:ye, xs:xe]
    
    # Save the cropped color image to visualize
    cv2.imwrite(str(OUTPUT_DIR / "manual_crop_color.png"), cropped)
    
    # 2. Resize to 640x640
    resized = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_CUBIC)
    
    # 3. Binarization
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save processed outputs
    cv2.imwrite(str(OUTPUT_DIR / "manual_crop_binarized.png"), binarized)
    print(f"Cropped and binarized images saved to {OUTPUT_DIR}")
    
    # 4. Run YOLO Inference
    model_path = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
    if model_path.exists():
        print("Running defect detection...")
        model = YOLO(str(model_path))
        
        # Convert to 3 channel for YOLO
        binarized_3ch = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(OUTPUT_DIR / "manual_crop_binarized_3ch.png"), binarized_3ch)
        
        results = model.predict(
            source=str(OUTPUT_DIR / "manual_crop_binarized_3ch.png"),
            imgsz=640,
            conf=0.25,
            save=True,
            project=str(OUTPUT_DIR),
            name="manual_inference",
            exist_ok=True
        )
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append((r.names[cls], round(conf, 3)))
                
        print("\n========== Detection Results on Manual Crop ==========")
        print(f"Total defects detected: {len(detections)}")
        for d in detections:
            print(f"- {d[0]} (confidence: {d[1]})")
        print("======================================================")
    else:
        print(f"YOLO model weights not found at {model_path}")

if __name__ == "__main__":
    main()
