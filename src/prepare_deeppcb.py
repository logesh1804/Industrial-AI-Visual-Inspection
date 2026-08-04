from pathlib import Path
from shutil import copy2
import random
from PIL import Image

# Change this only if your extracted dataset is in another location.
SOURCE_ROOT = Path(
    r"C:\Users\sabarishclean\Downloads\DeepPCB-master\DeepPCB-master\PCBData"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

# DeepPCB class IDs: 1-6
# YOLO class IDs:    0-5
CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "pin_hole",
]


def find_image(relative_path):
    """Find either a normal image or the *_test version."""
    expected_path = SOURCE_ROOT / relative_path

    if expected_path.exists():
        return expected_path

    test_image = expected_path.with_name(
        f"{expected_path.stem}_test{expected_path.suffix}"
    )

    if test_image.exists():
        return test_image

    raise FileNotFoundError(f"Image not found: {expected_path}")


def convert_label(source_label, output_label, image_width, image_height):
    """Convert DeepPCB x1 y1 x2 y2 class labels into YOLO labels."""
    yolo_lines = []

    for line in source_label.read_text().splitlines():
        x1, y1, x2, y2, class_id = map(float, line.split())

        # DeepPCB uses class IDs 1 to 6. YOLO uses 0 to 5.
        yolo_class_id = int(class_id) - 1

        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height

        yolo_lines.append(
            f"{yolo_class_id} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    output_label.write_text("\n".join(yolo_lines))


def prepare_split(pairs, split_name):
    image_folder = DATASET_ROOT / "images" / split_name
    label_folder = DATASET_ROOT / "labels" / split_name

    image_folder.mkdir(parents=True, exist_ok=True)
    label_folder.mkdir(parents=True, exist_ok=True)

    for image_relative, label_relative in pairs:
        source_image = find_image(Path(image_relative))
        source_label = SOURCE_ROOT / label_relative

        # Add group name so every output filename is unique.
        group_name = Path(image_relative).parts[0]
        output_name = f"{group_name}_{source_image.name}"

        output_image = image_folder / output_name
        output_label = label_folder / f"{Path(output_name).stem}.txt"

        copy2(source_image, output_image)

        with Image.open(source_image) as image:
            image_width, image_height = image.size

        convert_label(
            source_label,
            output_label,
            image_width,
            image_height,
        )

    print(f"{split_name}: {len(pairs)} images prepared")


def read_pairs(filename):
    pairs = []

    for line in (SOURCE_ROOT / filename).read_text().splitlines():
        image_relative, label_relative = line.split()
        pairs.append((image_relative, label_relative))

    return pairs


trainval_pairs = read_pairs("trainval.txt")
test_pairs = read_pairs("test.txt")

# Create a repeatable 80% training / 20% validation split.
random.Random(42).shuffle(trainval_pairs)
split_index = int(len(trainval_pairs) * 0.8)

prepare_split(trainval_pairs[:split_index], "train")
prepare_split(trainval_pairs[split_index:], "val")
prepare_split(test_pairs, "test")

print("DeepPCB conversion completed.")