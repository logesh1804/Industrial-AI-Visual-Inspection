import shutil
from pathlib import Path

source_dir = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection\output\phase2c_1_4hole_dynamic_tiled")
dest_dir = Path(r"C:\Users\sabarishclean\.gemini\antigravity-ide\brain\e1001de1-7b9a-49a0-b0fb-c022f925ab3c")

files_to_copy = [
    "01_original_camera.jpg",
    "02_distance_check.jpg",
    "03_four_hole_detection.jpg",
    "04_scale_reference.jpg",
    "05_pcb_roi.jpg",
    "06_registered_pcb.jpg",
    "07_trace_channel.jpg",
    "07_silkscreen_text_mask.jpg",
    "07_binarized_pcb.jpg",
    "08_tile_grid.jpg",
    "10_tile_detections.jpg",
    "11_reconstructed_tile_grid.jpg",
    "12_final_pcb_detection.jpg",
    "phase2c_1_engineering_report.jpg",
    "comparison_report.txt",
    "detections.csv"
]

dest_dir.mkdir(parents=True, exist_ok=True)
for filename in files_to_copy:
    src_file = source_dir / filename
    dest_file = dest_dir / filename
    if src_file.exists():
        shutil.copy(src_file, dest_file)
        print(f"Copied {filename} to artifacts.")
    else:
        print(f"File {filename} does not exist in source directory.")

# Copy defect crops
src_defects = source_dir / "13_defect_crops"
dst_defects = dest_dir / "13_defect_crops"
dst_defects.mkdir(parents=True, exist_ok=True)
for f in src_defects.glob("*.jpg"):
    shutil.copy(f, dst_defects / f.name)
