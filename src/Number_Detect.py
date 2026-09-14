import time
import cv2 as cv  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import easyocr as ocr  # type: ignore[reportMissingImports]
import os
from config import OCR_CONFIDENCE, OCR_GPU, WATCH_FOLDER

class NumberDetection:
    """
    Handles all OCR logic for reading the face value from a cropped die image.
    
    Usage:
        detector = NumberDetector()
        number, confidence = detector.read(crop)
    """

    def __init__(self):
        print("Loading EasyOCR...")
        self.reader = ocr.Reader(['en'], gpu=OCR_GPU)
        print("EasyOCR ready!")
        self.incoming_dir = WATCH_FOLDER

    def read(self, crop):
        """
        Takes a cropped die face image and returns the number showing.
        
        Args:
            crop: NumPy array (BGR image) of the die face
            
        Returns:
            tuple: (number as string, confidence as float)
                   (None, 0.0) if no number could be read
        """
        preprocessed = self._preprocess(crop)
        return self._run_ocr(preprocessed)
    
    def _preprocess(self, crop):
        # Upscale
        scale = max(1, 200 // min(crop.shape[:2]))
        crop  = cv.resize(crop, None, fx=scale, fy=scale,
                       interpolation=cv.INTER_CUBIC)

        # HSV color scale
        hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)

        # Saturation'
        saturation = hsv[:, :, 1]

        #  removes die texture noise
        blur = cv.GaussianBlur(saturation, (5, 5), 0)

        # Otsu threshold on the blurred image
        _, thresh = cv.threshold(blur, 30, 255,
                              cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        #inversion of image coloration
        inverted = cv.bitwise_not(thresh)

        # clean up noise
        kernel = np.ones((3, 3), np.uint8)
        clean  = cv.morphologyEx(inverted, cv.MORPH_OPEN, kernel)

        return clean


    def _run_ocr(self, image):
        """Runs EasyOCR on a preprocessed image and returns the best result."""

        results = self.reader.readtext(
            image,
            allowlist='0123456789',
            detail=1
        )

        if not results:
            return None, 0.0

        # Take highest confidence result
        best       = max(results, key=lambda x: x[2])
        text       = best[1]
        confidence = best[2]

        if confidence < OCR_CONFIDENCE:
            return None, confidence

        return text, confidence
    

    def watch_folder(self, poll_interval: float = 0.5):
        """Continuously watch incoming folder and OCR any new images."""
        processed = set()

        print(f"Watching '{self.incoming_dir}' for new images...")

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        while True:
            files = [f for f in os.listdir(self.incoming_dir)
                 if os.path.splitext(f)[1].lower() in image_exts]

            for filename in sorted(files):
                if filename in processed:
                    continue

                path = os.path.join(self.incoming_dir, filename)
                img = cv.imread(path)
                if img is None:
                    continue

                number, confidence = self.read(img)

                if number:
                    print(f"{filename:40s} -> {number}  (conf: {confidence:.0%})")
                else:
                    print(f"{filename:40s} -> No number detected  (conf: {confidence:.0%})")

                processed.add(filename)

            time.sleep(poll_interval)