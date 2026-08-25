import cv2
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
OUT_DIR = PROJECT_ROOT / "output" / "phase2a_live_reproducibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if cap.isOpened():
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        save_path = OUT_DIR / "current_webcam_frame.jpg"
        cv2.imwrite(str(save_path), frame)
        print(f"[SUCCESS] Captured frame from Webcam 0: {frame.shape} -> {save_path}")
    else:
        print("[ERROR] Could not read frame from Webcam 0.")
else:
    print("[ERROR] Could not open Webcam 0.")
