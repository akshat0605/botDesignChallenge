# KRITI Vision System

KRITI Vision System is an embedded computer vision platform designed to run on a **Raspberry Pi 4B**.  
It performs **real-time visual recognition** using a webcam and displays results through a **browser-based dashboard**.

The system combines **face recognition, object classification, and vehicle number plate recognition** into a single lightweight pipeline.

---

## Features

### Real-Time Video Processing
Processes live webcam feed directly on the Raspberry Pi.

### Face Recognition
Uses **InsightFace** embeddings and a stored face database to identify known individuals.

### Object Classification
A custom **YOLO classification model** detects multiple objects such as logos, furniture, pets, and vehicles.

### Vehicle Number Plate Recognition
Detects license plates and extracts text using **EasyOCR**.

Includes:
- multi-frame voting
- OCR stabilization
- noise filtering

### Web-Based Dashboard
The system hosts a browser interface showing:

- live camera feed
- detection results
- recognition confidence
- detection history
- system console logs
- number plate reader output

### Adjustable Detection Thresholds
Detection confidence thresholds can be tuned live using sliders in the dashboard.

---


### Pipeline Overview

1. **Frame Capture**
   - Frames are continuously captured from the webcam using OpenCV.

2. **Face Recognition**
   - Faces are detected and embedded using InsightFace.
   - Embeddings are compared against the stored `face_database.npy`.

3. **Object Classification**
   - A YOLO classification model identifies objects such as vehicles, logos, furniture, and pets.

4. **Plate Detection + OCR**
   - A YOLO detection model locates vehicle number plates.
   - Detected plates are cropped and passed to EasyOCR for text extraction.

5. **Result Aggregation**
   - Detected entities are filtered and aggregated before being displayed.

6. **Flask Dashboard**
   - The processed results are streamed to a web interface showing:
   - live camera feed
   - detection results
   - console logs
   - number plate recognition output

## Project Structure

```
kriti-vision/
│
├── app1.py                # Main application
├── best.pt                # YOLO classification model
├── plate.pt               # Plate detection model
├── face_database.npy      # Stored face embeddings
│
├── templates/
│   └── index.html         # Web interface
```

---

## Hardware Requirements

- Raspberry Pi 4B
- USB webcam
- MicroSD card (32GB recommended)

---

## Software Requirements

- Python 3.9+
- Raspberry Pi OS

Required Python libraries:

```
opencv-python
numpy
flask
ultralytics
easyocr
insightface
onnxruntime
```

---




## Detection Categories

### Face Recognition
- keenu-reeves  
- rodger-federer  
- henry-cavil  

### Brand Logos
- tesla  
- maybach  
- apple  
- keus  

### Furniture
- chair  
- dining table set  

### Vehicles
- car  
- motorbike  
- bicycle

### Pets
- dogs  
- cats  

### Other Objects
- parcel box  

- smart console  
- QR code  

---
