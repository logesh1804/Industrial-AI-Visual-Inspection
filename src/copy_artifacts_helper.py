import shutil
from pathlib import Path

source_dir = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection\output\phase2c_tiled_experiment")
dest_dir = Path(r"C:\Users\sabarishclean\.gemini\antigravity-ide\brain\e1d57a09-80d1-463e-aa6f-ef6f02d3e203")

files_to_copy = [
    "01_original_camera.jpg",
    "02_pcb_roi.jpg",
    "03_scale_reference.jpg",
    "04_aligned_pcb.jpg",
    "05_red_channel.jpg",
    "06_baseline_whole_pcb_640.jpg",
    "07_tile_grid_visualization.jpg",
    "08_tiled_final_detections.jpg",
    "09_full_pcb_detection_result.jpg",
    "comparison_report.txt",
    "detections.csv"
]

for filename in files_to_copy:
    src_file = source_dir / filename
    dest_file = dest_dir / filename
    if src_file.exists():
        shutil.copy(src_file, dest_file)
        print(f"Copied {filename} to artifacts.")
    else:
        print(f"File {filename} does not exist in source directory.")
