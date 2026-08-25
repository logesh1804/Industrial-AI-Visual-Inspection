"""
Capture live frames from IP camera on the local network.
"""
import urllib.request
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
OUT_DIR = PROJECT_ROOT / "output" / "phase2a_live_reproducibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

candidate_urls = [
    "http://192.168.1.50:8080/video",
    "http://192.168.1.50:8080/shot.jpg",
    "http://192.168.1.49:8080/video",
    "http://192.168.1.51:8080/video",
    "http://192.168.1.52:8080/video",
    "http://192.168.1.100:8080/video",
]

# Scan for any active IP on 192.168.1.X:8080
for last in range(1, 100):
    candidate_urls.append(f"http://192.168.1.{last}:8080/video")
    candidate_urls.append(f"http://192.168.1.{last}:8080/shot.jpg")
    candidate_urls.append(f"http://192.168.1.{last}:4747/video")

found_frame = False
active_url = None

for url in candidate_urls:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=0.3) as stream:
            if "shot.jpg" in url:
                img_bytes = stream.read()
                frame = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    active_url = url
                    found_frame = True
                    save_path = OUT_DIR / "live_ip_camera_raw_capture.jpg"
                    cv2.imwrite(str(save_path), frame)
                    print(f"[SUCCESS] Captured frame from {url}! Shape: {frame.shape} -> {save_path}")
                    break
            else:
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
                            active_url = url
                            found_frame = True
                            save_path = OUT_DIR / "live_ip_camera_raw_capture.jpg"
                            cv2.imwrite(str(save_path), frame)
                            print(f"[SUCCESS] Captured frame from {url}! Shape: {frame.shape} -> {save_path}")
                            break
                if found_frame:
                    break
    except Exception:
        pass

if not found_frame:
    print("[ERROR] Could not connect to IP Camera on scanned endpoints.")
