"""
MobileNet SSD Object Detection – Windows-Friendly Version
- Downloads official prototxt and caffemodel
- Stores them in a 'models' folder
- Uses UTF-8 for printing to avoid emoji errors
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # Fix Unicode printing

import cv2
import numpy as np
import os
import urllib.request
import ssl

# ====== Configuration ======
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Official URLs from chuanqi305/MobileNet-SSD
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
MODEL_URL    = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

PROTOTXT_NAME = "MobileNetSSD_deploy.prototxt"
MODEL_NAME    = "MobileNetSSD_deploy.caffemodel"

prototxt_path = os.path.join(MODELS_DIR, PROTOTXT_NAME)
model_path    = os.path.join(MODELS_DIR, MODEL_NAME)

# ====== Helper: download with progress ======
def download_file(url, dest):
    """Download a file with a simple progress indicator."""
    print(f"Downloading {os.path.basename(dest)} ...")
    try:
        # Bypass SSL certificate verification if needed
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

# ====== Check and download missing files ======
print("=" * 60)
print("MobileNet-SSD Model Check")
print("=" * 60)
print(f"Model directory: {MODELS_DIR}")

if not os.path.exists(prototxt_path):
    print("Prototxt not found. Downloading...")
    if not download_file(PROTOTXT_URL, prototxt_path):
        print("Failed to download prototxt. Exiting.")
        sys.exit(1)
else:
    print("Prototxt exists.")

if not os.path.exists(model_path):
    print("Model file not found. Downloading...")
    if not download_file(MODEL_URL, model_path):
        print("Failed to download model. Exiting.")
        sys.exit(1)
else:
    print("Model exists.")

# ====== Load the network ======
print("\nLoading MobileNet SSD model...")
try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("   • Make sure the prototxt and caffemodel are a matching pair.")
    print("   • Delete the files in 'models/' and rerun to download fresh copies.")
    sys.exit(1)

# ====== Class labels (20 classes + background) ======
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# ====== Open webcam ======
print("\nOpening webcam (press 'q' to quit)...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    (h, w) = frame.shape[:2]

    # Preprocess: resize to 300x300, scale, subtract mean
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                  0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    cv2.imshow("MobileNet SSD Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== Cleanup ======
cap.release()
cv2.destroyAllWindows()
print("Exited.")