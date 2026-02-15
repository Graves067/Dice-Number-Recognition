from Obj_Detect import DiceDetector
from Number_Detect import NumberDetection


def main():
    detector = DiceDetector(
        model_path="runs/detect/dice_detector9/weights/best.pt",
        confidence=0.6
    )
    detector.start_camera(width=640, height=480)
    detector.run()


if __name__ == '__main__':
    main()