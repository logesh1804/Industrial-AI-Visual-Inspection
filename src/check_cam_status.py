"""
Script to check IP camera status and capture both boards.
"""
import urllib.request
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
OUT_DIR = PROJECT_ROOT / "output" / "phase2b_real_pair_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IP_CAMERA_URL = "http://192.168.1.50:8080/video"

def test_stream():
    try:
        req = urllib.request.Request(IP_CAMERA_URL)
        with urllib.request.urlopen(req, timeout=3) as stream:
            bytes_data = bytes()
            for _ in range(30):
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
                        return True, frame
        return False, None
    except Exception as e:
        return False, str(e)

ok, result = test_stream()
if ok:
    print(f"[SUCCESS] IP Camera active. Frame shape: {result.shape}")
    cv2.imwrite(str(OUT_DIR / "test_live_frame.jpg"), result)
else:
    print(f"[STATUS] IP Camera currently not connected: {result}")
