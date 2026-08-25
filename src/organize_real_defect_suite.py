"""
Organize Photorealistic AI Generated PCB Defect Images & Real Camera Samples
into 'test_defective_pcbs_for_camera' and update the interactive viewer.
"""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
OUTPUT_DIR = PROJECT_ROOT / "test_defective_pcbs_for_camera"
ARTIFACTS_DIR = Path(r"C:\Users\sabarishclean\.gemini\antigravity-ide\brain\6aad780b-3d13-4fd6-9afc-fe2036ce7abb")
USER_UPLOAD_DIR = ARTIFACTS_DIR / ".user_uploaded"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Gather all generated photorealistic defect images
generated_files = list(ARTIFACTS_DIR.glob("pcb_defect_*.jpg"))
user_uploaded_files = list(USER_UPLOAD_DIR.glob("*.jpg"))

all_test_cards = []

# Copy AI Photorealistic Defect Images
for idx, gfile in enumerate(generated_files):
    dest = OUTPUT_DIR / f"real_ai_defect_{idx+1:02d}_{gfile.stem}.jpg"
    shutil.copy(gfile, dest)
    all_test_cards.append({
        "filename": dest.name,
        "title": f"Photorealistic Defect {idx+1:02d}",
        "defect": gfile.stem.replace("pcb_defect_", "").replace("_", " ").upper(),
        "category": "Photorealistic Industrial Defect"
    })

# Copy User Uploaded Real Defect Images
for idx, ufile in enumerate(user_uploaded_files):
    dest = OUTPUT_DIR / f"real_camera_sample_{idx+1:02d}.jpg"
    shutil.copy(ufile, dest)
    all_test_cards.append({
        "filename": dest.name,
        "title": f"Real Physical PCB Defect {idx+1:02d}",
        "defect": "PHYSICAL BURNT HOLE / SEVERED COPPER",
        "category": "Real Physical Camera Defect"
    })

# Existing Real PCB defect samples
existing_samples = [
    ("camera_defective_sample.png", "Real Green PCB Multiple Trace Defects"),
    ("pcb_color_defected.jpg", "Real Multi-Color PCB Defects")
]
for rname, rdesc in existing_samples:
    src = PROJECT_ROOT / "test_images" / rname
    if src.exists():
        dest = OUTPUT_DIR / f"physical_{rname}"
        shutil.copy(src, dest)
        all_test_cards.append({
            "filename": dest.name,
            "title": rname,
            "defect": rdesc,
            "category": "Physical Production Board"
        })

# Generate HTML viewer
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photorealistic Defective PCB Camera Test Suite</title>
    <style>
        body {{
            background: #0f172a;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 24px;
            text-align: center;
        }}
        h1 {{ color: #38bdf8; font-size: 26px; margin-bottom: 6px; }}
        p {{ color: #94a3b8; font-size: 14px; margin-top: 0; }}
        .instructions {{
            background: #1e293b;
            border-left: 4px solid #38bdf8;
            padding: 14px 18px;
            max-width: 720px;
            margin: 0 auto 24px auto;
            text-align: left;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.6;
        }}
        .card-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 24px;
        }}
        .pcb-card {{
            background: #1e293b;
            border-radius: 10px;
            padding: 14px;
            width: 360px;
            border: 1px solid #334155;
            box-shadow: 0 6px 16px rgba(0,0,0,0.4);
            transition: transform 0.2s, border-color 0.2s;
        }}
        .pcb-card:hover {{
            transform: translateY(-4px);
            border-color: #38bdf8;
        }}
        .pcb-card img {{
            width: 100%;
            height: 260px;
            object-fit: cover;
            background: #000;
            border-radius: 6px;
            cursor: pointer;
        }}
        .pcb-title {{
            font-weight: 600;
            color: #f8fafc;
            margin-top: 12px;
            font-size: 16px;
        }}
        .pcb-desc {{
            color: #fbbf24;
            font-size: 13px;
            font-weight: 500;
            margin-top: 4px;
        }}
        .badge {{
            display: inline-block;
            background: #0284c7;
            color: #fff;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin-top: 6px;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <h1>Real & Photorealistic Defective PCB Camera Test Suite</h1>
    <p>High-resolution physical macro images with authentic industrial PCB manufacturing defects</p>
    
    <div class="instructions">
        <strong>How to Test via Live Camera:</strong><br>
        1. Open this file on your <strong>phone screen or computer monitor</strong>.<br>
        2. Click on any image to view it full-screen.<br>
        3. Point your camera at the screen (~125 mm distance) and run <code>python src/phase2c_1_tiled_inspection.py</code>.<br>
        4. Press <strong>'S'</strong> to capture and verify real-time defect detection!
    </div>

    <div class="card-container">
"""

for card in all_test_cards:
    html_content += f"""        <div class="pcb-card">
            <a href="{card['filename']}" target="_blank">
                <img src="{card['filename']}" alt="{card['title']}">
            </a>
            <div class="pcb-title">{card['title']}</div>
            <div class="pcb-desc">{card['defect']}</div>
            <div class="badge">{card['category']}</div>
        </div>
"""

html_content += """    </div>
</body>
</html>
"""

html_path = OUTPUT_DIR / "VIEW_DEFECTS_ON_PHONE.html"
html_path.write_text(html_content, encoding="utf-8")
print(f"[SUCCESS] Compiled {len(all_test_cards)} photorealistic defect images into:")
print(f"  {html_path}")
