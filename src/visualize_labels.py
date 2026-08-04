from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = next((PROJECT_ROOT / "dataset/images/train").glob("*.jpg"))

LABEL_PATH = (
    PROJECT_ROOT / "dataset/labels/train" / f"{IMAGE_PATH.stem}.txt"
)

CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "pin_hole",
]

image = cv2.imread(str(IMAGE_PATH))
height, width = image.shape[:2]

for line in LABEL_PATH.read_text().splitlines():
    class_id, cx, cy, box_w, box_h = map(float, line.split())

    x1 = int((cx - box_w / 2) * width)
    y1 = int((cy - box_h / 2) * height)
    x2 = int((cx + box_w / 2) * width)
    y2 = int((cy + box_h / 2) * height)

    label = CLASS_NAMES[int(class_id)]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image, label, (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    )

output_path = PROJECT_ROOT / "output/label_check.jpg"
cv2.imwrite(str(output_path), image)

print(f"Saved label check image: {output_path}")