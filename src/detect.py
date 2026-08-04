from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# Run inference on a sample image using GPU 0
results = model(
    "https://ultralytics.com/images/bus.jpg",
    device=0,
    save=True,
    project="output",
    name="first_inference"
)

# Print detected object names and confidence scores
for result in results:
    for class_id, confidence in zip(result.boxes.cls, result.boxes.conf):
        object_name = result.names[int(class_id)]
        print(f"{object_name}: {float(confidence):.2f}")