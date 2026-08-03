import cv2
from ultralytics import YOLO

print("=" * 40)
print("Industrial AI Visual Inspection")
print("=" * 40)

print("OpenCV Version:", cv2.__version__)

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

print("YOLO loaded successfully!")
print("Ready for AI detection.")
