import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import time

# Balloon colour detection (HSV)
def detect_colour(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colour_ranges = {
        "pink":   [(140, 50, 50), (170, 255, 255)],
        "yellow": [(20, 100, 100), (30, 255, 255)],
        "blue":   [(90, 100, 100), (130, 255, 255)],
        "white":  [(0, 0, 200), (180, 40, 255)],
        "black":  [(0, 0, 0), (180, 255, 50)]
    }
    max_pixels = 0
    detected_colour = "unknown"
    for colour, (lower, upper) in colour_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        count = cv2.countNonZero(mask)
        if count > max_pixels:
            max_pixels = count
            detected_colour = colour
    return detected_colour

# Position detection
def get_position(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < 2 * frame_width / 3:
        return "center"
    else:
        return "right"

# Draw modern bounding box (outline + label background)
def draw_fancy_bbox(img, x1, y1, x2, y2, label, color=(0, 255, 0)):
    # Outline box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label background
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), color, -1)

    # Label text
    cv2.putText(img, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return img

# Class colors
class_colors = {
    "Balloon": (255, 0, 255),
    "Hammer": (0, 255, 0),
    "Traffic_Cone": (255, 140, 0),
    "Tennis_Ball": (0, 255, 255)
}

# Main detection loop using `requests` for MJPEG streaming
def run_detection(model_path, source, resolution):
    model = YOLO(model_path)

    # Open MJPEG stream using requests
    stream = requests.get(source, stream=True)

    if stream.status_code != 200:
        print(f"Error: Unable to connect to video source {source}")
        return

    bytes_data = b''  # Temporary buffer to store MJPEG bytes

    # Class IDs for your 4-class model
    all_classes = [0, 1, 2, 3]  # Balloon, Hammer, Traffic_Cone, Tennis_Ball
    balloon_class = [0]         # Balloon only

    while True:
        # Read the MJPEG stream in chunks
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # Start of JPEG image
            b = bytes_data.find(b'\xff\xd9')  # End of JPEG image

            if a != -1 and b != -1:
                jpg_data = bytes_data[a:b+2]  # Extract the complete JPEG image
                bytes_data = bytes_data[b+2:]  # Keep the remaining data

                # Convert the JPEG byte data into an image
                img = Image.open(BytesIO(jpg_data))
                img_np = np.array(img)

                # Convert to BGR (OpenCV format)
                frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # High confidence detections
                results_main = model(frame, conf=0.7, classes=all_classes)

                # Lower confidence pass for balloons (for colour detection)
                results_balloon = model(frame, conf=0.65, classes=balloon_class)

                frame_h, frame_w = frame.shape[:2]

                # Draw high-confidence detections
                for r in results_main:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x_center = (x1 + x2) // 2
                        position = get_position(x_center, frame_w)

                        # Balloon → detect colour
                        if label.lower() == "balloon":
                            cropped = frame[y1:y2, x1:x2]
                            balloon_colour = detect_colour(cropped)
                            label = f"{balloon_colour} balloon"

                        label = f"{label} ({position}) {conf:.2f}"
                        color = class_colors.get(model.names[cls], (0, 255, 0))
                        frame = draw_fancy_bbox(frame, x1, y1, x2, y2, label, color)

                # Draw extra balloons from low-confidence pass
                for r in results_balloon:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf >= 0.85:  # Already shown above
                            continue
                        cls = int(box.cls[0])
                        label = model.names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x_center = (x1 + x2) // 2
                        position = get_position(x_center, frame_w)

                        cropped = frame[y1:y2, x1:x2]
                        balloon_colour = detect_colour(cropped)
                        label = f"{balloon_colour} balloon (low conf) ({position}) {conf:.2f}"

                        frame = draw_fancy_bbox(frame, x1, y1, x2, y2, label, (200, 0, 200))

                # Display the frame with detections
                cv2.imshow("YOLO Detection", frame)

                # Exit the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

# Run from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--source", type=str, required=True, help="'webcam' or path to video/image")
    parser.add_argument("--resolution", nargs=2, type=int, help="Resolution width height")
    args = parser.parse_args()

    run_detection(args.model, args.source, tuple(args.resolution) if args.resolution else None)
