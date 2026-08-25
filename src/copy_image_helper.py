import shutil
from pathlib import Path

source = Path(r"C:\Users\sabarishclean\.gemini\antigravity-ide\brain\f2938bf0-7817-4f7f-bd77-e2777e4d9229\colored_defected_pcb_1787224853698.jpg")
destination = Path(r"c:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection\test_images\pcb_color_defected.jpg")

if source.exists():
    shutil.copy(source, destination)
    print(f"Successfully copied image to {destination}")
else:
    print(f"Error: Source image not found at {source}")
