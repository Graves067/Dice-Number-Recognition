from Obj_Detect import DiceDetector
from Number_Detect import NumberDetection
import threading

def main():
    number_detector = NumberDetection()

    # Start OCR folder watcher in background thread
    ocr_thread = threading.Thread(target=number_detector.watch_folder, daemon=True)
    ocr_thread.start()

    # Live detection — saves crops to src\Incoming
    detector = DiceDetector(
        model_path="runs/detect/dice_detector9/weights/best.pt",
        confidence=0.6
    )
    detector.start_camera(width=640, height=480, device=1)
    detector.run()


if __name__ == '__main__':
    main()