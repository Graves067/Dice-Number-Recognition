import cv2 as cv
import numpy as np
import easyocr as ocr
from config import OCR_CONFIDENCE, OCR_GPU

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