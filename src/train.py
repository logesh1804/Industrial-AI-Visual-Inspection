from pathlib import Path
from ultralytics import YOLO


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    model = YOLO(PROJECT_ROOT / "yolov8n.pt")

    model.train(
        data=PROJECT_ROOT / "dataset" / "pcb.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,          # safer on Windows
        project=PROJECT_ROOT / "output" / "training",
        name="pcb_defect_yolov8n",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()