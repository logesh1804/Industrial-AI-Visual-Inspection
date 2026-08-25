"""
Lightweight Neural Text Detector Model Downloader
Fetches pre-trained PP-OCRv3 ONNX model (~3MB) for real-time text detection.
"""
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\sabarishclean\Desktop\Industrial-AI-Visual-Inspection")
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URLS = [
    (
        "SWHL RapidOCR PP-OCRv3 (HuggingFace)",
        "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/ch_PP-OCRv3_det_infer.onnx",
        MODELS_DIR / "text_detection_db.onnx"
    ),
    (
        "gqfwqgw PaddleOCR ONNX (HuggingFace)",
        "https://huggingface.co/gqfwqgw/paddle-ocr/resolve/main/ch_PP-OCRv3_det_infer.onnx",
        MODELS_DIR / "text_detection_db.onnx"
    ),
    (
        "Monter PP-OCRv3 (HuggingFace)",
        "https://huggingface.co/monter/ppocr_v3/resolve/main/ch_PP-OCRv3_det_infer.onnx",
        MODELS_DIR / "text_detection_db.onnx"
    )
]

def download_model():
    print("=" * 70)
    print("DOWNLOADING LIGHTWEIGHT NEURAL TEXT DETECTOR (ONNX)")
    print("=" * 70)
    
    target_path = MODELS_DIR / "text_detection_db.onnx"
    if target_path.exists() and target_path.stat().st_size > 100_000:
        print(f"[OK] Model already exists at: {target_path} ({target_path.stat().st_size / 1e6:.2f} MB)")
        return True
        
    for name, url, dest in MODEL_URLS:
        print(f"\nAttempting download from: {name}...")
        print(f"URL: {url}")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(dest, 'wb') as out_file:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                block_size = 65536
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        sys.stdout.write(f"\rDownloading... {percent:.1f}% ({downloaded / 1e6:.2f} MB)")
                        sys.stdout.flush()
                        
            if dest.exists() and dest.stat().st_size > 100_000:
                print(f"\n[SUCCESS] Downloaded {name} -> {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
                return True
            else:
                print("\n[WARN] Downloaded file is too small or incomplete.")
        except Exception as e:
            print(f"\n[WARN] Failed to download from {name}: {e}")
            
    print("\n[ERROR] All automated download sources failed.")
    return False

if __name__ == "__main__":
    download_model()
