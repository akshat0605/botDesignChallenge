import cv2
import threading
import time
import numpy as np
import easyocr
import re
from collections import deque, Counter

from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from insightface.app import FaceAnalysis

app = Flask(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
FACE_THRESHOLD   = 0.45
OBJECT_THRESHOLD = 0.35
TOP_K            = 5

# ── Plate ─────────────────────────────────────────────────────────────────────
PLATE_HISTORY   = deque(maxlen=5)
PLATE_MIN_VOTES = 1
PLATE_HOLD      = 40

# ── Pi capture resolution ─────────────────────────────────────────────────────
CAPTURE_W = 640
CAPTURE_H = 480

# ── Category map ──────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    'keenu-reeves'         : 'Face recognition',
    'rodger-federer'       : 'Face recognition',
    'henry-cavil'          : 'Face recognition',

    'tesla'                : 'brand logo',
    'maybach'              : 'brand logo',
    'keus'                 : 'brand logo',
    'apple'                : 'brand logo',

    'chair'                : 'furniture',
    'dining table set'     : 'furniture',

    'parcel box'           : 'parcel box',

    'dogs'                 : 'pets',
    'cats'                 : 'pets',

    'iit guwhati'          : 'qr code',
    'smart console'        : 'smart switch',

    'vehicle number plate' : 'vehicle number plate',
    'Vehicle number plates': 'vehicle number plate',

    'motorbike'            : 'vehicle',
    'car'                  : 'vehicle',
    'bicycle'              : 'vehicle',
}

# ── Indian plate regex ────────────────────────────────────────────────────────
PLATE_REGEX = re.compile(
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$'
    r'|^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models…")

object_model = YOLO("best.pt")
plate_model  = YOLO("plate.pt")

face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=0, det_size=(160, 160))

face_database = np.load("face_database.npy", allow_pickle=True).item()

# EasyOCR — gpu=False for Pi, loaded once at startup
ocr_reader = easyocr.Reader(['en'], gpu=False)

print("All models ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def recognize_face(embedding):
    embedding = embedding / np.linalg.norm(embedding)
    best_name, best_score = None, 0.0
    for name, emb_list in face_database.items():
        sims = []
        for db_emb in emb_list:
            db_emb = db_emb / np.linalg.norm(db_emb)
            sims.append(float(np.dot(embedding, db_emb)))
        if sims:
            score = float(np.mean(sims))
            if score > best_score:
                best_score, best_name = score, name
    if best_score > FACE_THRESHOLD:
        return best_name, best_score
    return None, None


def classify_objects(frame):
    detections = []
    results = object_model(frame, imgsz=160, verbose=False)[0]
    if not results.probs:
        return detections
    probs       = results.probs.data.cpu().numpy()
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    for cls_id in top_indices:
        conf = float(probs[cls_id])
        if conf > OBJECT_THRESHOLD:
            name = object_model.names[cls_id]
            detections.append({
                "name"    : name,
                "category": CATEGORY_MAP.get(name, "object"),
                "conf"    : int(conf * 100),
            })
    return detections


def clean_plate(text):
    text = re.sub(r'IND', '', text.upper())
    text = re.sub(r'[^A-Z0-9]', '', text)
    if len(text) >= 6:
        c   = list(text)
        d2l = {'0':'O','1':'I','2':'Z','5':'S','8':'B'}
        l2d = {'O':'0','I':'1','Z':'2','S':'5','B':'8'}
        for i in range(min(2, len(c))):
            c[i] = d2l.get(c[i], c[i])
        for i in range(2, min(4, len(c))):
            c[i] = l2d.get(c[i], c[i])
        for i in range(max(0, len(c) - 4), len(c)):
            c[i] = l2d.get(c[i], c[i])
        text = "".join(c)
    return text


def valid_plate(text):
    return bool(PLATE_REGEX.match(text))


def ocr_crop(crop):
    """Single fast EasyOCR pass — returns best effort immediately."""
    h, w = crop.shape[:2]
    if w != 200:
        crop = cv2.resize(crop, (200, max(1, int(h * 200 / w))),
                          interpolation=cv2.INTER_LINEAR)
    results = ocr_reader.readtext(
        crop,
        detail=0,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
    )
    return clean_plate(" ".join(results))


# ── Camera ────────────────────────────────────────────────────────────────────

class Camera:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame   = None
        self.objects = []
        self.plate   = None

        self.running       = True
        self.frame_counter = 0
        self.plate_timer   = 0

        self._ocr_lock    = threading.Lock()
        self._plate_crop  = None
        self._ocr_result  = None
        self._ocr_busy    = False   # True while OCR thread is working
        self.plate_reading = False  # exposed to API — "trying to read"

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop,  daemon=True).start()
        threading.Thread(target=self._ocr_loop,     daemon=True).start()

    # ── Thread 1: capture ─────────────────────────────────────────────────────
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.005)

    # ── Thread 2: YOLO + face — never blocked by OCR ─────────────────────────
    def _detect_loop(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.02)
                continue

            frame_copy = self.frame.copy()
            detections = []

            faces = face_app.get(frame_copy)
            if faces:
                name, score = recognize_face(faces[0].embedding)
                if name:
                    detections.append({
                        "name"    : name,
                        "category": CATEGORY_MAP.get(name, "Face recognition"),
                        "conf"    : int(score * 100),
                    })

            detections += classify_objects(frame_copy)

            # If a face was recognized, remove all non-face detections
            # Prevents false object hits (chair, dog etc) showing alongside a person
            if any(d["category"] == "Face recognition" for d in detections):
                detections = [d for d in detections if d["category"] == "Face recognition"]

            self.objects = detections
            self.frame_counter += 1

            if self.plate_timer > 0:
                self.plate_timer -= 1

            plate_visible = any(
                d["name"].lower().replace(" ", "") in
                ("vehiclenumberplate", "vehiclenumberplates",
                 "numberplate", "licenseplate")
                for d in detections
            )

            if plate_visible:
                self.plate_reading = True   # tell frontend we're trying
                with self._ocr_lock:
                    self._plate_crop = frame_copy
                with self._ocr_lock:
                    result = self._ocr_result
                    if result:
                        self._ocr_result = None
                if result:
                    self.plate         = result
                    self.plate_reading = False
                    self.plate_timer   = PLATE_HOLD
            else:
                self.plate_reading = False
                if self.plate_timer == 0:
                    self.plate = None

    # ── Thread 3: OCR — fully independent, never slows detection ─────────────
    def _ocr_loop(self):
        while self.running:
            crop = None
            with self._ocr_lock:
                if self._plate_crop is not None:
                    crop             = self._plate_crop
                    self._plate_crop = None

            if crop is None:
                time.sleep(0.01)
                continue

            # Skip running plate_model again — object classifier already confirmed
            # a plate is in frame. Just run Tesseract on the full frame directly.
            # This removes one entire YOLO inference per OCR cycle (~100-200ms saved).
            raw = ocr_crop(crop)
            print(f"[OCR] '{raw}' valid={valid_plate(raw)}")

            if len(raw) < 4:
                continue

            # Cap at 10 characters and show instantly — no validation gate
            raw = raw[:10]

            PLATE_HISTORY.append(raw)
            best, votes = Counter(PLATE_HISTORY).most_common(1)[0]

            # Always push best result immediately
            with self._ocr_lock:
                self._ocr_result = best

    # ── JPEG for stream ───────────────────────────────────────────────────────
    def get_jpeg(self):
        if self.frame is None:
            return None
        _, jpeg = cv2.imencode('.jpg', self.frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes()


cam = Camera()


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            frame = cam.get_jpeg()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def detections():
    return jsonify({
        "objects"      : cam.objects or [],
        "plate_reading": cam.plate_reading,
        "plate"        : {
            "number"  : cam.plate,
            "category": "vehicle number plate",
        } if cam.plate else None,
    })


@app.route('/set_thresholds')
def set_thresholds():
    global FACE_THRESHOLD, OBJECT_THRESHOLD
    FACE_THRESHOLD   = float(request.args.get('face',   45)) / 100
    OBJECT_THRESHOLD = float(request.args.get('object', 35)) / 100
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)