"""
Camera Focus Tuner Tool
------------------------
Allows live tuning of Auto-Focus and Manual Focus settings for USB cameras in OpenCV.
Features:
  - Toggle Auto-Focus ON / OFF (Key 'A')
  - Manual Focus Slider Trackbar (0 - 255)
  - Zoom-in Region of Interest (ROI) box to inspect PCB sharpness in real-time
  - Sharpness score (Laplacian variance) displayed in real time
"""

import cv2
import numpy as np

def calculate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main():
    print("==================================================")
    print("           CAMERA FOCUS TUNER TOOL                ")
    print("==================================================")
    
    # Try USB camera indices
    cap = None
    cam_idx = None
    for idx in [1, 2, 0]:
        test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not test_cap.isOpened():
            test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            cap = test_cap
            cam_idx = idx
            break
            
    if cap is None:
        print("[ERROR] No USB camera found!")
        return

    # Set high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Camera Focus Tuner (OpenCV)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    # Initial autofocus state
    autofocus = 0  # Start manual to test slider
    current_focus = 40

    # Try setting initial manual focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, current_focus)

    def on_focus_trackbar(val):
        nonlocal current_focus, autofocus
        current_focus = val
        if autofocus == 0:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, val)

    def on_autofocus_trackbar(val):
        nonlocal autofocus
        autofocus = val
        cap.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
        if autofocus == 0:
            cap.set(cv2.CAP_PROP_FOCUS, current_focus)

    cv2.createTrackbar("Auto-Focus (0=Off, 1=On)", window_name, 0, 1, on_autofocus_trackbar)
    cv2.createTrackbar("Manual Focus Val", window_name, current_focus, 255, on_focus_trackbar)

    print("\nControls:")
    print("  [A]     : Toggle Auto-Focus ON/OFF")
    print("  [+] / [-] (or [ / ]): Increment / Decrement Manual Focus")
    print("  [Q/ESC] : Quit")
    print("--------------------------------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        
        # Center crop for sharpness calculation & zoomed preview
        cw, ch = 300, 300
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - cw//2), max(0, cy - ch//2)
        x2, y2 = min(w, cx + cw//2), min(h, cy + ch//2)
        center_roi = frame[y1:y2, x1:x2]

        sharpness = calculate_sharpness(center_roi)

        # Draw UI on frame
        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(display, "Focus Target Box", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Info overlay banner
        cv2.rectangle(display, (0, 0), (w, 55), (20, 20, 20), -1)
        af_text = "ENABLED (Auto)" if autofocus == 1 else f"DISABLED (Manual Val: {current_focus})"
        cv2.putText(display, f"Auto-Focus: {af_text}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if autofocus==1 else (0, 200, 255), 2)
        
        # Sharpness score (Higher = Sharper)
        sharp_color = (0, 255, 0) if sharpness > 100 else ((0, 165, 255) if sharpness > 40 else (0, 0, 255))
        cv2.putText(display, f"Sharpness Score: {sharpness:.1f} (Higher is sharper)", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sharp_color, 2)

        # Show Zoomed Inset in top right corner
        zoom_view = cv2.resize(center_roi, (220, 220), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(zoom_view, (0, 0), (220, 220), (0, 255, 255), 2)
        cv2.putText(zoom_view, "2x Zoomed", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        display[60:280, w-230:w-10] = zoom_view

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('a') or key == ord('A'):
            autofocus = 1 - autofocus
            cv2.setTrackbarPos("Auto-Focus (0=Off, 1=On)", window_name, autofocus)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
            if autofocus == 0:
                cap.set(cv2.CAP_PROP_FOCUS, current_focus)
        elif key == ord('+') or key == ord('=') or key == ord(']'):
            current_focus = min(255, current_focus + 5)
            cv2.setTrackbarPos("Manual Focus Val", window_name, current_focus)
            if autofocus == 0:
                cap.set(cv2.CAP_PROP_FOCUS, current_focus)
        elif key == ord('-') or key == ord('_') or key == ord('['):
            current_focus = max(0, current_focus - 5)
            cv2.setTrackbarPos("Manual Focus Val", window_name, current_focus)
            if autofocus == 0:
                cap.set(cv2.CAP_PROP_FOCUS, current_focus)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Best focus value found: {current_focus} (Auto-Focus: {autofocus})")

if __name__ == "__main__":
    main()
