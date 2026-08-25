"""
Live Camera Capture & Multi-Modality Validation Script
"""
import sys
import os
import time
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PT      = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
OUT_DIR      = PROJECT_ROOT / "output" / "phase2a_live_reproducibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IP_CAMERA_URL = "http://192.168.1.50:8080/video"

def get_live_frame(camera_source=0):
    """
    Attempts to read from IP Camera URL first; if unreachable, falls back to Webcam index 0.
    """
    if IP_CAMERA_URL:
        try:
            req = urllib.request.Request(IP_CAMERA_URL)
            stream = urllib.request.urlopen(req, timeout=2)
            bytes_data = bytes()
            for _ in range(50):
                chunk = stream.read(4096)
                if not chunk:
                    break
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame, "IP_Camera"
        except Exception:
            pass

    # Fallback to local webcam
    cap = cv2.VideoCapture(camera_source)
    if cap.isOpened():
        # Read a few frames to let auto-exposure settle
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return True, frame, f"Webcam_{camera_source}"
            
    return False, None, "None"

print("Live capture module ready.")
