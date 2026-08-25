import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = PROJECT_ROOT / "captured_images"
OUTPUT_DIR = PROJECT_ROOT / "output" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def order_points(pts):
    """Order coordinates as: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_and_warp_pcb(image_path, output_size=640):
    """Detects PCB, warps perspective to a square, and binarizes it."""
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    
    orig = img.copy()
    
    # Step 1: Pre-processing for contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection or adaptive thresholding
    # Depending on background, cv2.Canny or thresholding works best
    edged = cv2.Canny(blurred, 50, 150)
    
    # Step 2: Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pcb_contour = None
    max_area = 0
    
    # Find the largest rectangular-ish contour
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10000:  # Minimum area threshold
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # If the contour has 4 points, it's likely our rectangular PCB
            if len(approx) == 4:
                if area > max_area:
                    pcb_contour = approx
                    max_area = area

    # Fallback: if no 4-point contour is found, use the largest bounding box
    if pcb_contour is None and len(contours) > 0:
        print("Warning: Could not detect 4-corner contour. Falling back to largest bounding box.")
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        pcb_contour = pts.reshape(-1, 1, 2)

    if pcb_contour is None:
        print("Error: No PCB contour found.")
        return None, None

    # Step 3: Perspective Warp
    pts = pcb_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Warp to a square destination
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (output_size, output_size))
    
    # Step 4: Binarization (Convert warped color image to black and white)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding or Adaptive thresholding
    # DeepPCB uses binary thresholding where background is black, traces are white (or vice versa)
    # Let's apply Otsu thresholding
    _, binarized = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: invert binary image if background of DeepPCB is black and captured is white (or vice versa)
    # Check if we need to invert based on trace characteristics
    # For now, let's keep standard thresholding
    
    return warped, binarized

def main():
    # Find any captured image to test
    captured_files = list(CAPTURE_DIR.glob("*.png"))
    if not captured_files:
        print(f"No captured images found in {CAPTURE_DIR}")
        return
    
    test_img = captured_files[0]
    print(f"Processing test image: {test_img.name}")
    
    warped, binarized = crop_and_warp_pcb(test_img)
    
    if warped is not None:
        warped_path = OUTPUT_DIR / "warped_color.png"
        binarized_path = OUTPUT_DIR / "binarized_640.png"
        
        cv2.imwrite(str(warped_path), warped)
        cv2.imwrite(str(binarized_path), binarized)
        print(f"Processed images saved to {OUTPUT_DIR}")
        
        # Load YOLO model and run inference on the binarized image
        model_path = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
        if not model_path.exists():
            print(f"YOLO model weights not found at {model_path}")
            print("Please make sure you have trained a model or placed weights there.")
            return
            
        print(f"Loading YOLO model from {model_path.name}...")
        model = YOLO(str(model_path))
        
        # YOLO prediction on binarized image
        # DeepPCB binary images are grayscale, but YOLOv8 expects 3 channels.
        # cv2.threshold outputs single channel. We need to convert it to 3-channel (BGR) for YOLO.
        binarized_3ch = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
        binarized_3ch_path = OUTPUT_DIR / "binarized_640_3ch.png"
        cv2.imwrite(str(binarized_3ch_path), binarized_3ch)
        
        results = model.predict(
            source=str(binarized_3ch_path),
            imgsz=640,
            conf=0.25,
            save=True,
            project=str(OUTPUT_DIR),
            name="inference_results",
            exist_ok=True
        )
        
        # Parse detections
        detections = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    "class": names[cls],
                    "confidence": round(conf, 3)
                })
                
        print("\n========== Detection Results on Preprocessed Crop ==========")
        print(f"Total defects detected: {len(detections)}")
        for d in detections:
            print(f"- {d['class']} (confidence: {d['confidence']})")
        print("=============================================================")
        print(f"Visualized predictions saved in {OUTPUT_DIR}/inference_results")
        
    else:
        print("Processing failed.")

if __name__ == "__main__":
    from ultralytics import YOLO
    main()
