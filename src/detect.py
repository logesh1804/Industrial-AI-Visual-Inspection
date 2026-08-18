from pathlib import Path
from ultralytics import YOLO


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    model = YOLO(
        PROJECT_ROOT
        / "output"
        / "training"
        / "pcb_defect_yolov8n"
        / "weights"
        / "best.pt"
    )

    results = model.predict(
        source=PROJECT_ROOT / "test_images",
        imgsz=640,
        conf=0.25,
        save=True,
        show=False,
    )

    print("\nDetection completed!")
    print("Results saved in:")
    print(PROJECT_ROOT / "runs" / "detect")


if __name__ == "__main__":
    main()