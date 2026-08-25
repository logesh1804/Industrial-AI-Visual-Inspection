import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "output" / "training" / "pcb_defect_yolov8n" / "weights" / "best.pt"
CAPTURE_DIR = PROJECT_ROOT / "captured_images"
OUTPUT_DIR = PROJECT_ROOT / "output" / "predictions"

CAPTURE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. CAMERA CONNECTION SETTINGS
# ==========================================
# If using a mobile IP camera app (e.g. IP Webcam, DroidCam, etc.), 
IP_CAMERA_URL = "http://192.168.1.44:8080/video" 

# ==========================================
# 2. SCREW HOLE DETECTION SETTINGS
# ==========================================
HOUGH_SETTINGS = {
    "minDist": 25,      # Min distance between circles
    "param1": 50,       # Canny threshold
    "param2": 35,       # Center detection threshold (higher = less sensitive, less noise)
    "minRadius": 6,     # Min screw hole radius (pixels)
    "maxRadius": 30     # Max screw hole radius (pixels)
}

# ==========================================
# helper functions
# ==========================================
import urllib.request

class IPStreamReader:
    def __init__(self, url):
        self.url = url
        # Use urllib to open connection with a timeout
        self.stream = urllib.request.urlopen(url, timeout=5)
        self.bytes_data = bytes()
        
    def read(self):
        """Reads stream buffer and decodes the next complete JPEG frame"""
        start_time = time.time()
        try:
            while time.time() - start_time < 3.0:
                chunk = self.stream.read(4096)
                if not chunk:
                    return False, None
                self.bytes_data += chunk
                a = self.bytes_data.find(b'\xff\xd8')  # JPEG Start
                b = self.bytes_data.find(b'\xff\xd9')  # JPEG End
                if a != -1 and b != -1:
                    if a < b:
                        jpg_bytes = self.bytes_data[a:b+2]
                        self.bytes_data = self.bytes_data[b+2:]
                        if len(jpg_bytes) > 0:
                            frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                return True, frame
                    else:
                        # Discard everything up to the start marker to align with the next clean frame
                        self.bytes_data = self.bytes_data[a:]
        except Exception as e:
            print(f"Warning: Stream read error or timeout: {e}")
            return False, None
        return False, None

    def release(self):
        self.stream.close()

def connect_camera(ip_url):
    """Attempts to connect to cameras in priority order: IP Cam -> USB Cam -> Laptop Cam"""
    # 1. Try IP Cam via custom robust MJPEG reader
    if ip_url and ip_url.strip():
        print(f"Priority 1: Connecting to IP Camera at {ip_url}...")
        try:
            reader = IPStreamReader(ip_url)
            ret, frame = reader.read()
            if ret and frame is not None:
                print("Successfully connected to IP Camera via Custom MJPEG Reader!")
                return reader, True
        except Exception as e:
            print(f"Failed to connect to IP Camera: {e}. Moving to USB cameras.")

    # 2. Try USB Web Cams (Windows/Linux indices 1 and 2)
    for idx in [1, 2]:
        print(f"Priority 2: Connecting to USB Web Camera (Index {idx})...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # CAP_DSHOW is faster on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Successfully connected to USB Camera (Index {idx})!")
            return cap, False
            
    # 3. Try Laptop Cam (Index 0)
    print("Priority 3: Connecting to Laptop Camera (Index 0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Successfully connected to Laptop Camera!")
        return cap, False

    return None, False

def order_points(pts):
    """Orders 4 coordinates as: top-left, top-right, bottom-right, bottom-left"""
    # Sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    
    # Grab the left-most and right-most points from the sorted x-coordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    
    # Now, sort the left-most coordinates according to their y-coordinates
    # so we can grab the top-left and bottom-left points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    
    # Now, sort the right-most coordinates according to their y-coordinates
    # so we can grab the top-right and bottom-right points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    
    return np.array([tl, tr, br, bl], dtype="float32")

def find_pcb_contour(frame):
    """Detects the largest rectangular/quadrilateral contour of the PCB"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive edge detection
    edged = cv2.Canny(blurred, 30, 130)
    
    # Dilate/erode to close gaps in trace borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pcb_contour = None
    max_area = 0
    frame_area = frame.shape[0] * frame.shape[1]
    
    for c in contours:
        area = cv2.contourArea(c)
        # The PCB must be reasonably large in the frame (at least 2% of frame area)
        if area > (frame_area * 0.02): 
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Look for 4-corner quadrilateral (the PCB)
            if len(approx) == 4:
                if area > max_area:
                    pcb_contour = approx
                    max_area = area
                    
    # Fallback to largest bounding box contour if no perfect 4-corner shape found
    if pcb_contour is None and len(contours) > 0:
        large_contours = [c for c in contours if cv2.contourArea(c) > (frame_area * 0.02)]
        if large_contours:
            largest = max(large_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            pcb_contour = pts.reshape(-1, 1, 2)
            
    return pcb_contour

def warp_pcb(frame, contour):
    """Warps the perspective of the detected PCB contour, preserving actual aspect ratio"""
    pts = contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate physical width and height to preserve aspect ratio
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    return warped

# ==========================================
# main execution
# ==========================================
def main():
    # Load YOLO model
    if MODEL_PATH.exists():
        print(f"Loading YOLO model from: {MODEL_PATH.name}")
        model = YOLO(MODEL_PATH)
    else:
        print(f"Warning: YOLO weights not found at {MODEL_PATH}. Defect detection will be unavailable.")
        model = None

    # Connect to camera
    cap, is_ip = connect_camera(IP_CAMERA_URL)
    if cap is None:
        print("Error: Could not connect to any camera.")
        return

    # If it is a USB or Laptop Camera, set it to HD resolution (1280x720) for better detail
    if not is_ip:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Set camera resolution to 1280x720")

    # Try to read the first frame. Network cameras can have a slight delay before frames are available.
    frame = None
    print("Initializing camera feed...")
    for attempt in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        print(f"Waiting for camera stream frame... (Attempt {attempt+1}/15)")
        time.sleep(0.2)
        
    if frame is None:
        print("Error: Could not grab frame from camera stream.")
        cap.release()
        return

    # Extract base URL for IP camera still snapshot if it's an IP webcam
    SNAPSHOT_URL = None
    if is_ip:
        base_url = IP_CAMERA_URL.rsplit('/', 1)[0]
        SNAPSHOT_URL = f"{base_url}/shot.jpg"
        print(f"IP Camera Snapshot URL: {SNAPSHOT_URL}")

    print("\n==============================================")
    print("      Dynamic PCB Inspection Dashboard        ")
    print("==============================================")
    print("S : Capture high-accuracy frame and run inspection")
    print("Q : Quit application\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Try to grab again in case of network glitch
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
        # Keep a clean copy of the frame (for high-accuracy capture later)
        clean_frame = frame.copy()
        
        # Detect PCB boundary
        pcb_contour = find_pcb_contour(frame)
        
        if pcb_contour is not None:
            # Draw green boundary around the PCB
            cv2.drawContours(frame, [pcb_contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, "PCB Detected", (pcb_contour[0][0][0], pcb_contour[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Detect screw holes only inside the bounding box of the PCB contour
            x, y, w_box, h_box = cv2.boundingRect(pcb_contour)
            pcb_roi = frame[y:y+h_box, x:x+w_box]
            
            if pcb_roi.size > 0:
                gray_roi = cv2.cvtColor(pcb_roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.medianBlur(gray_roi, 5)
                
                # Scale Hough parameters dynamically based on current frame width relative to 640px
                scale_factor = frame.shape[1] / 640.0
                min_dist = int(HOUGH_SETTINGS["minDist"] * scale_factor)
                min_radius = int(HOUGH_SETTINGS["minRadius"] * scale_factor)
                max_radius = int(HOUGH_SETTINGS["maxRadius"] * scale_factor)
                
                circles = cv2.HoughCircles(
                    blurred_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=min_dist,
                    param1=HOUGH_SETTINGS["param1"],
                    param2=HOUGH_SETTINGS["param2"],
                    minRadius=min_radius,
                    maxRadius=max_radius
                )
                
                # Draw red screw holes on the live display frame
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        cx = i[0] + x
                        cy = i[1] + y
                        radius = i[2]
                        # Draw circle
                        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 2)
                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 3)
        else:
            # If no PCB is detected, warn the user on screen
            cv2.putText(frame, "PCB NOT DETECTED - Adjust Position", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Show live visual feedback frame (resizing only for display, keeping frame full-res)
        display_w = 960
        display_h = int(frame.shape[0] * (display_w / frame.shape[1]))
        resized_display = cv2.resize(frame, (display_w, display_h))
        cv2.imshow("Industrial Visual Inspection - Live UI", resized_display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            print("\nCapturing high-accuracy clean frame...")
            
            high_res_frame = None
            if SNAPSHOT_URL:
                import urllib.request
                print(f"Requesting full-resolution photo from: {SNAPSHOT_URL}")
                try:
                    resp = urllib.request.urlopen(SNAPSHOT_URL, timeout=8)
                    image_bytes = np.asarray(bytearray(resp.read()), dtype="uint8")
                    high_res_frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    print(f"Successfully loaded snapshot: {high_res_frame.shape[1]}x{high_res_frame.shape[0]}")
                except Exception as e:
                    print(f"Failed to fetch snapshot: {e}. Falling back to video stream frame.")
                    high_res_frame = clean_frame.copy()
            else:
                high_res_frame = clean_frame.copy()
                
            # Perform high-accuracy contour detection on the clean full-resolution still frame
            clean_contour = find_pcb_contour(high_res_frame)
            
            # Fallback: Scale live coordinates to fit the high-resolution frame size
            if clean_contour is None and pcb_contour is not None:
                print("PCB not directly detected on still photo. Scaling live coordinates...")
                h_live, w_live = clean_frame.shape[:2]
                h_hr, w_hr = high_res_frame.shape[:2]
                scale_x = w_hr / w_live
                scale_y = h_hr / h_live
                
                scaled_contour = pcb_contour.copy().astype(np.float32)
                scaled_contour[:, :, 0][:, 0] *= scale_x
                scaled_contour[:, :, 1][:, 0] *= scale_y
                clean_contour = scaled_contour.astype(np.int32)
            
            if clean_contour is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # Warp to flat board (preserves true aspect ratio)
                warped = warp_pcb(high_res_frame, clean_contour)
                
                # Save color warped crop
                color_crop_path = CAPTURE_DIR / f"high_res_crop_{timestamp}.png"
                cv2.imwrite(str(color_crop_path), warped)
                
                # Binarize to match DeepPCB style (black traces on white background)
                # Extract Red channel (channel 2 in BGR) where green mask is dark and copper/solder is bright
                red_channel = warped[:, :, 2]
                
                # Apply Otsu global thresholding to get solid black traces on white background
                _, binarized = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Auto-invert if binarized image is mostly black (background substrate should be white)
                white_pixels = np.sum(binarized == 255)
                total_pixels = binarized.size
                white_ratio = white_pixels / total_pixels
                if white_ratio < 0.5:
                    print(f"Auto-inverting threshold: binarized image was mostly black ({white_ratio:.1%} white pixels)")
                    binarized = cv2.bitwise_not(binarized)
                
                # Resize the binarized image to 640x640 for the YOLO model input
                binarized_resized = cv2.resize(binarized, (640, 640), interpolation=cv2.INTER_CUBIC)
                
                # Convert to 3-channel for YOLO input
                binarized_3ch = cv2.cvtColor(binarized_resized, cv2.COLOR_GRAY2BGR)
                binarized_path = CAPTURE_DIR / f"high_res_binarized_{timestamp}.png"
                cv2.imwrite(str(binarized_path), binarized_3ch)
                
                print(f"Saved preprocessed crop: {binarized_path.name}")
                
                # Run Defect Model
                if model is not None:
                    print("Performing defect inspection...")
                    results = model.predict(
                        source=str(binarized_path),
                        imgsz=640,
                        conf=0.25,
                        save=True,
                        project=str(OUTPUT_DIR),
                        name="live_dynamic_inference",
                        exist_ok=True
                    )
                    
                    total_defects = 0
                    for r in results:
                        total_defects += len(r.boxes)
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            print(f"- Detected Defect: {r.names[cls]} (conf: {round(conf,2)})")
                            
                    status = "PASS" if total_defects == 0 else "FAIL"
                    print("--------------------------------")
                    print("Inspection Result")
                    print("--------------------------------")
                    print("Detected Defects :", total_defects)
                    print("Status :", status)
                    print("--------------------------------")
                else:
                    print("YOLO model unavailable.")
            else:
                print("Error: High-accuracy capture failed because PCB border was not detected in clean frame.")
                print("Please position the PCB clearly in front of the camera.")
                
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
