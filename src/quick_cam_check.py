import urllib.request
import cv2
import numpy as np

IP_URL = "http://192.168.1.50:8080/video"

try:
    req = urllib.request.Request(IP_URL)
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
                    print(f"[SUCCESS] IP Camera online! Frame shape: {frame.shape}")
                    cv2.imwrite("output/test_live_check.jpg", frame)
                    exit(0)
    print("[ERROR] Stream opened but no frame decoded.")
except Exception as e:
    print(f"[ERROR] Cannot connect to {IP_URL}: {e}")
