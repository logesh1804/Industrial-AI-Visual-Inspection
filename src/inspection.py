from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import json
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def inspect_pcb(image_path):

    model = YOLO(
        PROJECT_ROOT /
        "output" /
        "training" /
        "pcb_defect_yolov8n" /
        "weights" /
        "best.pt"
    )

    start = time.time()

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        save=True
    )

    end = time.time()

    detections = []

    for result in results:

        names = result.names

        for box in result.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            detections.append(
                {
                    "class": names[cls],
                    "confidence": round(conf, 3)
                }
            )

    counts = Counter(d["class"] for d in detections)

    total = len(detections)

    status = "PASS" if total == 0 else "FAIL"

    report = {

        "image": Path(image_path).name,

        "inspection_time": round(end - start, 3),

        "total_defects": total,

        "status": status,

        "defects": dict(counts)

    }

    print("\n========== PCB Inspection Report ==========")

    print(json.dumps(report, indent=4))

    return report


if __name__ == "__main__":

    image = PROJECT_ROOT / "test_images" / "pcb_test.png"

    inspect_pcb(image)