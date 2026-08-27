#!/usr/bin/env python3
"""
Jetson Nano Hardware, Camera & AI Diagnostics Script
Inspects Jetson Tegra release, CUDA GPU acceleration, CSI/USB Cameras, and YOLO inference.
"""
import sys
import os
import platform
import time
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_jetson_environment():
    print_section("1. SYSTEM & JETSON HARDWARE CHECK")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Check Tegra Release
    tegra_file = Path("/etc/nv_tegra_release")
    if tegra_file.exists():
        print(f"[OK] Jetson Tegra Release: {tegra_file.read_text().strip()}")
    else:
        print("[INFO] Not running on native Tegra hardware or /etc/nv_tegra_release not present.")

    # Check Device Tree Model
    model_file = Path("/sys/firmware/devicetree/base/model")
    if model_file.exists():
        try:
            print(f"[OK] Device Model: {model_file.read_text().strip()}")
        except Exception:
            pass

def check_cuda_pytorch():
    print_section("2. PYTORCH & CUDA ACCELERATION CHECK")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"[OK] GPU Device Name: {torch.cuda.get_device_name(0)}")
            print(f"[OK] Device Count: {torch.cuda.device_count()}")
            # Quick tensor test
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print(f"[OK] GPU Tensor Compute Verified: matrix multiplication passed.")
        else:
            print("[WARN] CUDA is not available in PyTorch. Inference will run on CPU.")
    except ImportError:
        print("[FAIL] PyTorch is not installed.")

def get_jetson_csi_pipeline(sensor_id=0, width=1280, height=720, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=1"
    )

def check_cameras():
    print_section("3. CAMERA HARDWARE CHECK (CSI & USB)")
    
    # 1. Test CSI Camera
    print("[1] Testing Jetson CSI Camera (GStreamer nvarguscamerasrc)...")
    try:
        pipeline = get_jetson_csi_pipeline(sensor_id=0, width=1280, height=720, framerate=30)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[SUCCESS] Jetson CSI Camera is active! Frame shape: {frame.shape}")
            else:
                print("[WARN] CSI Camera opened but failed to grab frame.")
            cap.release()
        else:
            print("[INFO] Jetson CSI Camera not opened.")
    except Exception as e:
        print(f"[INFO] CSI Camera probe: {e}")

    # 2. Test USB Cameras
    for idx in range(3):
        print(f"[2] Testing USB Camera index {idx}...")
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[SUCCESS] Camera index {idx} connected! Frame shape: {frame.shape}")
            else:
                print(f"[WARN] Camera index {idx} opened but could not read frame.")
            cap.release()
        else:
            print(f"[INFO] Camera index {idx} not available.")

def check_yolo_model():
    print_section("4. YOLO MODEL & INFERENCE CHECK")
    model_path = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "yolov8n.pt"
    
    if not model_path.exists():
        print(f"[FAIL] No model weights found at {model_path}")
        return

    print(f"Loading Model: {model_path}...")
    try:
        from ultralytics import YOLO
        import numpy as np
        model = YOLO(str(model_path))
        # Test inference on dummy 640x640 frame
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        t0 = time.time()
        res = model.predict(dummy, verbose=False)
        dt = (time.time() - t0) * 1000
        print(f"[SUCCESS] Model inference passed in {dt:.1f}ms!")
    except Exception as e:
        print(f"[FAIL] YOLO model inference test failed: {e}")

if __name__ == "__main__":
    print_section("JETSON NANO DIAGNOSTICS & VERIFICATION SUITE")
    check_jetson_environment()
    check_cuda_pytorch()
    check_cameras()
    check_yolo_model()
    print_section("DIAGNOSTICS COMPLETE")
