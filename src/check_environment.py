import torch
import cv2
from ultralytics import YOLO

print("=" * 50)
print("Industrial AI Visual Inspection")
print("=" * 50)

print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")


if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version    : {torch.version.cuda}")

print(f"OpenCV Version  : {cv2.__version__}")

model = YOLO("yolov8n.pt")
print("YOLO Model      : Loaded Successfully")

print("=" * 50)
print("Environment Ready!")
print("=" * 50)
