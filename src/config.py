# Paths
MODEL_PATH   = r"runs/detect/dice_detector9/weights/best.pt"
WATCH_FOLDER = r"src\Incoming"   # must match DiceDetector.save_dir in Obj_Detect.py
DATASET_DIR  = r"src\Dataset"    # labeled crops land here, sorted by face value

# Detection settings
CONFIDENCE     = 0.6    # YOLO confidence 
PADDING        = 20     #  padding for around bounding box

# OCR settings
OCR_CONFIDENCE = 0.5    # minimum OCR confidence 
OCR_GPU        = True  # GPU accelerated OCR

