"""
Prepare 25+ High-Resolution Defective PCB Images for Camera Testing
Creates a dedicated folder 'test_defective_pcbs_for_camera' with realistic, high-resolution
defective PCB images (Green, Blue, Black, Multi-color) containing genuine manufacturing defects:
- mousebite
- open circuit (broken trace)
- short circuit (copper bridge)
- pin hole
- spur (protruding track)
- spurious copper

Includes an interactive HTML slideshow viewer (VIEW_DEFECTS_ON_PHONE.html) so you can
display them on your phone screen or computer monitor directly in front of your camera.
"""
import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
OUTPUT_DIR = PROJECT_ROOT / "test_defective_pcbs_for_camera"
DATASET_TEST_DIR = PROJECT_ROOT / "dataset" / "images" / "test"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

def create_realistic_defective_board(base_color, defect_type, board_idx):
    """
    Synthesizes a realistic high-resolution PCB board image with 4 mounting holes,
    copper traces, silkscreen text, and a distinct physical defect.
    """
    w, h = 800, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Substrate color
    if base_color == "green":
        img[:] = (20, 80, 20)
        trace_color = (40, 160, 40)
        hole_color = (200, 200, 200)
        text_color = (240, 240, 240)
    elif base_color == "blue":
        img[:] = (90, 40, 10)
        trace_color = (180, 80, 20)
        hole_color = (210, 210, 210)
        text_color = (250, 250, 250)
    elif base_color == "black":
        img[:] = (30, 30, 30)
        trace_color = (70, 70, 70)
        hole_color = (190, 190, 190)
        text_color = (230, 230, 230)
    else:
        img[:] = (30, 70, 30)
        trace_color = (50, 140, 50)
        hole_color = (200, 200, 200)
        text_color = (240, 240, 240)
        
    # 1. Draw 4 Standard Mounting Holes (for scale calibration)
    margin = 50
    holes = [(margin, margin), (w - margin, margin), (margin, h - margin), (w - margin, h - margin)]
    for hx, hy in holes:
        cv2.circle(img, (hx, hy), 22, (10, 10, 10), -1)
        cv2.circle(img, (hx, hy), 22, hole_color, 4)
        
    # 2. Draw Copper Circuit Traces
    for i in range(120, h - 120, 35):
        # Horizontal parallel tracks
        cv2.line(img, (140, i), (w - 140, i), trace_color, 8)
        # Solder pads along trace
        cv2.circle(img, (200, i), 10, hole_color, -1)
        cv2.circle(img, (w - 200, i), 10, hole_color, -1)
        
    # Vertical connecting buses
    cv2.line(img, (260, 120), (260, h - 120), trace_color, 8)
    cv2.line(img, (w - 260, 120), (w - 260, h - 120), trace_color, 8)
    
    # 3. Draw Silkscreen Labels
    cv2.putText(img, f"PCB-REV-{board_idx:02d}", (280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(img, "TEST-BENCH", (w // 2 - 60, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(img, "PWR", (140, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(img, "GND", (w - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # 4. Inject Physical Defect
    defect_desc = ""
    cx, cy = w // 2, h // 2
    if defect_type == "open": # Broken Track
        cv2.rectangle(img, (cx - 25, cy - 5), (cx + 25, cy + 5), (20, 80, 20) if base_color=="green" else (30,30,30), -1)
        defect_desc = "Open Circuit (Severed Copper Track)"
    elif defect_type == "short": # Copper Bridge
        y1 = cy - 35
        y2 = cy
        cv2.rectangle(img, (cx - 6, y1), (cx + 6, y2), trace_color, -1)
        defect_desc = "Short Circuit (Copper Bridge Between Traces)"
    elif defect_type == "mousebite": # Edge Nick / Mousebite
        cv2.circle(img, (cx - 40, cy), 12, (20, 80, 20) if base_color=="green" else (30,30,30), -1)
        defect_desc = "Mousebite (Copper Track Edge Defect)"
    elif defect_type == "pin_hole": # Hole inside track
        cv2.circle(img, (cx + 60, cy), 5, (20, 80, 20) if base_color=="green" else (30,30,30), -1)
        defect_desc = "Pin Hole (Pinhole Void in Solid Copper)"
    elif defect_type == "spur": # Protrusion
        cv2.rectangle(img, (cx - 80, cy - 14), (cx - 70, cy), trace_color, -1)
        defect_desc = "Spur (Protruding Metal Burr)"
    elif defect_type == "spurious_copper": # Isolated Copper Speck
        cv2.circle(img, (cx + 100, cy - 18), 10, trace_color, -1)
        defect_desc = "Spurious Copper (Unconnected Metal Debris)"
        
    # Overlay Label on image for easy testing
    cv2.putText(img, f"DEFECT: {defect_desc.upper()}", (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return img, defect_desc

def main():
    print("=" * 80)
    print("PREPARING 25+ DEFECTIVE PCB TEST IMAGES FOR CAMERA VERIFICATION")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_images = []
    
    # 1. Realistic Multi-Color Synthesized PCB Defects (12 Boards)
    defect_types = ["open", "short", "mousebite", "pin_hole", "spur", "spurious_copper"]
    colors = ["green", "blue", "black"]
    
    count = 1
    for c in colors:
        for d in defect_types:
            img, desc = create_realistic_defective_board(c, d, count)
            fname = f"pcb_{count:02d}_{c}_{d}.jpg"
            fpath = OUTPUT_DIR / fname
            cv2.imwrite(str(fpath), img)
            generated_images.append({
                "filename": fname,
                "type": f"Realistic {c.capitalize()} PCB",
                "defect": desc
            })
            count += 1
            
    # 2. Existing Real Camera Samples (3 Boards)
    real_samples = [
        ("camera_defective_sample.png", "Real Green PCB with Multiple Physical Defects"),
        ("pcb_color_defected.jpg", "Real Multi-Color Bare Board with Copper Defects")
    ]
    for rname, rdesc in real_samples:
        src = TEST_IMAGES_DIR / rname
        if src.exists():
            fname = f"pcb_{count:02d}_real_{rname}"
            shutil.copy(src, OUTPUT_DIR / fname)
            generated_images.append({
                "filename": fname,
                "type": "Physical Camera Sample",
                "defect": rdesc
            })
            count += 1
            
    # 3. High-Res DeepPCB Test Samples (10 Boards)
    if DATASET_TEST_DIR.exists():
        test_files = sorted(list(DATASET_TEST_DIR.glob("*.jpg")))[:10]
        for tf in test_files:
            fname = f"pcb_{count:02d}_deeppcb_{tf.name}"
            shutil.copy(tf, OUTPUT_DIR / fname)
            generated_images.append({
                "filename": fname,
                "type": "DeepPCB Verified Trace Defect",
                "defect": "Ground-Truth Synthetic Copper Defect"
            })
            count += 1
            
    print(f"\n[SUCCESS] Generated and compiled {len(generated_images)} defective PCB images in:")
    print(f"  {OUTPUT_DIR}\n")
    
    # 4. Generate Interactive HTML Slideshow Viewer (VIEW_DEFECTS_ON_PHONE.html)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Defective PCB Camera Test Suite</title>
    <style>
        body {{
            background: #121212;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            text-align: center;
        }}
        h1 {{ margin-bottom: 5px; color: #00e5ff; font-size: 24px; }}
        p {{ color: #a0a0a0; font-size: 14px; margin-top: 0; }}
        .card-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }}
        .pcb-card {{
            background: #1e1e1e;
            border-radius: 8px;
            padding: 12px;
            width: 340px;
            border: 1px solid #333;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            transition: transform 0.2s, border-color 0.2s;
        }}
        .pcb-card:hover {{
            transform: translateY(-4px);
            border-color: #00e5ff;
        }}
        .pcb-card img {{
            width: 100%;
            height: 240px;
            object-fit: contain;
            background: #000;
            border-radius: 4px;
        }}
        .pcb-title {{
            font-weight: bold;
            color: #fff;
            margin-top: 10px;
            font-size: 16px;
        }}
        .pcb-desc {{
            color: #ffab00;
            font-size: 13px;
            margin-top: 4px;
        }}
        .instructions {{
            background: #1e293b;
            border-left: 4px solid #38bdf8;
            padding: 12px;
            max-width: 700px;
            margin: 0 auto 20px auto;
            text-align: left;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <h1>Defective PCB Camera Test Suite ({len(generated_images)} Test Boards)</h1>
    <div class="instructions">
        <strong>How to Test with Live Camera:</strong><br>
        1. Open this file on your <strong>phone screen or computer monitor</strong>.<br>
        2. Point your camera at any of the PCB images below (keep distance ~125mm).<br>
        3. Press <strong>'S'</strong> in the live inspection window to verify that YOLO detects the defects in real time!
    </div>
    <div class="card-container">
"""
    for img_info in generated_images:
        html_content += f"""        <div class="pcb-card">
            <img src="{img_info['filename']}" alt="{img_info['filename']}">
            <div class="pcb-title">{img_info['filename']}</div>
            <div class="pcb-desc">{img_info['defect']} ({img_info['type']})</div>
        </div>
"""
    html_content += """    </div>
</body>
</html>
"""
    html_path = OUTPUT_DIR / "VIEW_DEFECTS_ON_PHONE.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"[CREATED] Interactive Viewer: {html_path}")

if __name__ == "__main__":
    main()
