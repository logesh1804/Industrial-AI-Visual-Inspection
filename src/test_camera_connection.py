"""
Script to test camera reachability (IP Camera URL and Local Webcams).
"""
import cv2
import urllib.request
import numpy as np
import time

IP_URL = "http://192.168.1.50:8080/video"

print("--- Testing IP Camera ---")
try:
    req = urllib.request.Request(IP_URL)
    stream = urllib.request.urlopen(req, timeout=3)
    print(f"[SUCCESS] Connected to IP Camera at {IP_URL}")
    bytes_data = bytes()
    success = False
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
                print(f"[SUCCESS] Captured frame from IP Camera! Shape: {frame.shape}")
                cv2.imwrite("test_images/live_ip_cam_test.jpg", frame)
                success = True
                break
    if not success:
        print("[WARNING] Connected to IP Camera but could not decode frame.")
except Exception as e:
    print(f"[FAILED] IP Camera at {IP_URL} not reachable: {e}")

print("\n--- Testing Local Webcams (Index 0, 1) ---")
for idx in [0, 1]:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[SUCCESS] Local Webcam index {idx} opened! Shape: {frame.shape}")
            cv2.imwrite(f"test_images/live_webcam_{idx}_test.jpg", frame)
        else:
            print(f"[INFO] Webcam index {idx} opened but could not read frame.")
        cap.release()
    else:
        print(f"[INFO] Webcam index {idx} not available.")
