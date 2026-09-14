import cv2 as cv
from ultralytics import YOLO
from Number_Detect import NumberDetection
import os
import time
from collections import defaultdict


class DiceDetector:
    def __init__(self, model_path: str, confidence: float = 0.6):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.camera = None
        self.number_detector = NumberDetection()
        self.save_dir = r"src\Incoming"
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

                number, ocr_conf = self.number_detector.read(crop)

                # is_new_object returns True the FIRST time we see a die,
                # so save on True (once per unique die position)
                if self.is_new_object(x1, y1, x2, y2):
                    self.save_detection(frame, crop, cls_name, confidence, number)

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{cls_name.upper()}  {confidence:.0%}"
                if number:
                    label += f" | {number} ({ocr_conf:.0%})"

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

    def save_detection(self, frame, crop, label, confidence, number=None):
        """Save cropped dice image — once per unique die (no cooldown needed since
        is_new_object already gates this to one call per die position)."""
        if confidence < self.confidence:
            return

        self.save_counters[label] += 1

        if number:
            filename = f"{label}_{number}_{self.save_counters[label]:03d}.jpg"
        else:
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