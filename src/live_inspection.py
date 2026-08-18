import cv2
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"

CAPTURE_DIR = PROJECT_ROOT / "captured_images"
OUTPUT_DIR = PROJECT_ROOT / "output" / "predictions"

CAPTURE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("\n==============================")
print(" Industrial PCB Inspection")
print("==============================")
print("S : Capture & Inspect")
print("Q : Quit\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Industrial PCB Inspection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        image_path = CAPTURE_DIR / f"pcb_{timestamp}.png"

        cv2.imwrite(str(image_path), frame)

        print(f"\nCaptured : {image_path.name}")

        results = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            save=True,
            project=str(OUTPUT_DIR),
            name="inspection",
            exist_ok=True
        )

        total_defects = 0

        for result in results:
            total_defects += len(result.boxes)

        status = "PASS" if total_defects == 0 else "FAIL"

        print("--------------------------------")

        print("Inspection Result")

        print("--------------------------------")

        print("Detected Defects :", total_defects)

        print("Status :", status)

        print("--------------------------------")

    elif key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()