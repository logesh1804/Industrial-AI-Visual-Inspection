import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
CAPTURE_DIR = PROJECT_ROOT / "captured_images"
OUTPUT_DIR = PROJECT_ROOT / "output" / "predictions"

CAPTURE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO model
if MODEL_PATH.exists():
    print(f"Loading YOLO model from: {MODEL_PATH.name}")
    model = YOLO(MODEL_PATH)
else:
    print(f"Warning: YOLO weights not found at {MODEL_PATH}. Defect detection will be unavailable.")
    model = None

# ==========================================
# ROI CALIBRATION: Adjust these to fit your camera setup
# ==========================================
# Bounding box coordinates on your live camera frame:
CROP_BOX = {
    "y_start": 120,
    "y_end": 360,
    "x_start": 160,
    "x_end": 480
}

# ==========================================
# SCREW HOLE DETECTION SETTINGS
# ==========================================
# Adjust these parameters if screw holes are not being detected or if there are false detections
HOUGH_SETTINGS = {
    "minDist": 30,      # Min distance between centers of detected circles
    "param1": 50,       # Upper threshold for internal Canny edge detector
    "param2": 25,       # Threshold for center detection (smaller = more circles, larger = more accurate)
    "minRadius": 8,     # Min radius of screw holes in pixels
    "maxRadius": 25     # Max radius of screw holes in pixels
}

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# Set resolution (common defaults)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n==============================================")
# Industrial PCB Inspection Dashboard
print("  Industrial PCB ROI Inspection Dashboard  ")
print("==============================================")
print("S : Crop, Binarize & Run YOLO Defect Inspection")
print("Q : Quit App\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break
        
    h, w, c = frame.shape
    
    # Ensure crop coordinates are within live frame bounds
    ys = max(0, min(h, CROP_BOX["y_start"]))
    ye = max(ys, min(h, CROP_BOX["y_end"]))
    xs = max(0, min(w, CROP_BOX["x_start"]))
    xe = max(xs, min(w, CROP_BOX["x_end"]))
    
    # 1. Extract Cropped PCB Region for Processing
    crop = frame[ys:ye, xs:xe]
    
    # 2. Detect Screw Holes inside Cropped Region
    if crop.size > 0:
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Apply median blur to reduce noise for circle detection
        blurred_crop = cv2.medianBlur(gray_crop, 5)
        
        # Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred_crop,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=HOUGH_SETTINGS["minDist"],
            param1=HOUGH_SETTINGS["param1"],
            param2=HOUGH_SETTINGS["param2"],
            minRadius=HOUGH_SETTINGS["minRadius"],
            maxRadius=HOUGH_SETTINGS["maxRadius"]
        )
        
        # Draw detected screw holes relative to full frame
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Map coordinates from crop space back to frame space
                center_x = i[0] + xs
                center_y = i[1] + ys
                radius = i[2]
                
                # Draw outer circle boundary in RED
                cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
                # Draw center point of circle in RED
                cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)
                
    # 3. Draw Green Boundary Lines (ROI Crop Box)
    cv2.rectangle(frame, (xs, ys), (xe, ye), (0, 255, 0), 2)
    
    # Add labels on the live frame
    cv2.putText(frame, "PCB Inspection ROI Boundary", (xs, ys - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
    # Display feed
    cv2.imshow("Industrial PCB Inspection System", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Handle key press 's' for inspection
    if key == ord('s'):
        if crop.size == 0:
            print("Error: Empty crop region.")
            continue
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        raw_crop_path = CAPTURE_DIR / f"roi_crop_{timestamp}.png"
        binarized_path = CAPTURE_DIR / f"roi_binarized_{timestamp}.png"
        
        # Save raw cropped color image
        cv2.imwrite(str(raw_crop_path), crop)
        
        # Resize to 640x640 (YOLO input shape)
        resized_crop = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
        
        # Binarization matching DeepPCB training format
        gray_resized = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)
        _, binarized = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert binary to 3-channel for YOLO inference
        binarized_3ch = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(binarized_path), binarized_3ch)
        
        print(f"\nCaptured and Preprocessed Crop: {binarized_path.name}")
        
        if model is not None:
            print("Running defect inspection...")
            results = model.predict(
                source=str(binarized_path),
                imgsz=640,
                conf=0.25,
                save=True,
                project=str(OUTPUT_DIR),
                name="live_roi_inference",
                exist_ok=True
            )
            
            total_defects = 0
            for r in results:
                total_defects += len(r.boxes)
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    print(f"- Detected Defect: {r.names[cls]} (conf: {round(conf,2)})")
            
            status = "PASS" if total_defects == 0 else "FAIL"
            print("--------------------------------")
            print("Inspection Result")
            print("--------------------------------")
            print("Detected Defects :", total_defects)
            print("Status :", status)
            print("--------------------------------")
        else:
            print("YOLO model not loaded. Defect inspection skipped.")
            
    elif key == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
