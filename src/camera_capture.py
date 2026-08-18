import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAVE_DIR = PROJECT_ROOT / "captured_images"
SAVE_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 's' to save image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("PCB Camera", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = SAVE_DIR / "pcb_capture.png"
        cv2.imwrite(str(filename), frame)
        print(f"Saved: {filename}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()