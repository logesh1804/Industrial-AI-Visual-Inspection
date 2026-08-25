"""
Script to capture fresh live frame from IP camera and save it.
"""
import urllib.request
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
OUT_DIR = PROJECT_ROOT / "output" / "phase2b_real_pair_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IP_CAMERA_URL = "http://192.168.1.50:8080/video"

try:
    req = urllib.request.Request(IP_CAMERA_URL)
    with urllib.request.urlopen(req, timeout=4) as stream:
        bytes_data = bytes()
        for _ in range(40):
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
                    save_path = OUT_DIR / "current_live_camera_frame.jpg"
                    cv2.imwrite(str(save_path), frame)
                    print(f"[SUCCESS] Captured current frame: {frame.shape} -> {save_path}")
                    break
except Exception as e:
    print(f"[ERROR] Could not read from IP camera: {e}")
