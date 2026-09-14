import cv2 as cv  # type: ignore[import-not-found]
from ultralytics import YOLO  # type: ignore[import-not-found]
import os
import time
from collections import defaultdict
from config import WATCH_FOLDER


class DiceDetector:
    """
    Detects dice in a live camera feed and saves a crop of each newly-seen
    die to `save_dir`. This class only detects and crops — it does not
    interpret the face value. Interpretation happens downstream, on the
    saved images in `save_dir` (see Number_Detect.watch_folder / the
    upcoming ResNet classifier), which keeps a single source of truth for
    face-value identification and lets the saved crops double as a growing
    labeled dataset.
    """

    def __init__(self, model_path: str, confidence: float = 0.6):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.camera = None
        self.save_dir = WATCH_FOLDER
        self.save_counters = defaultdict(int)
        self.last_saved = {}
        self.save_cooldown = 1.5
        self.seen_centers = []
        self.center_threshold = 40

        os.makedirs(self.save_dir, exist_ok=True)

    def start_camera(self, width: int = 640, height: int = 480, device: int = 1):
        self.camera = cv.VideoCapture(device)
        ret, _ = self.camera.read()
        if not ret:
            print(f"[WARN] Could not open device {device}, falling back to device 0")
            self.camera.release()
            self.camera = cv.VideoCapture(0)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    def stop_camera(self):
        if self.camera:
            self.camera.release()
        cv.destroyAllWindows()

    def annotate_frame(self, frame, results):
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = self.model.names[cls_id]
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # is_new_object returns True the FIRST time we see a die,
                # so save on True (once per unique die position). Face-value
                # interpretation happens downstream on the saved image.
                if self.is_new_object(x1, y1, x2, y2):
                    self.save_detection(frame, crop, cls_name, confidence)

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{cls_name.upper()}  {confidence:.0%}"

                cv.putText(frame, label, (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        return frame

    def run(self):
        if not self.camera or not self.camera.isOpened():
            raise RuntimeError("Camera is not started. Call start_camera() first.")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                results = self.model(frame, conf=self.confidence, verbose=False)
                frame = self.annotate_frame(frame, results)
                cv.imshow("Dice Detector", frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop_camera()

    def save_detection(self, frame, crop, label, confidence):
        """Save cropped dice image — once per unique die (no cooldown needed since
        is_new_object already gates this to one call per die position).

        Filename intentionally has no face-value label yet: that gets decided
        downstream (OCR today, ResNet classifier later) and, once labeled
        crops are needed for training, should be sorted into class folders
        rather than encoded in the filename here.
        """
        if confidence < self.confidence:
            return

        self.save_counters[label] += 1
        filename = f"{label}_{self.save_counters[label]:03d}.jpg"

        path = os.path.join(self.save_dir, filename)
        cv.imwrite(path, crop)
        print(f"Saved: {path}")

    def is_new_object(self, x1, y1, x2, y2):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        for px, py in self.seen_centers:
            if abs(cx - px) < self.center_threshold and abs(cy - py) < self.center_threshold:
                return False  # already seen this die

        self.seen_centers.append((cx, cy))
        return True  # new die