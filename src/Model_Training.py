from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=r"C:\Users\cagil\Documents\GitHub\Dice-Number-Recognition\Dice-2\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        name="dice_detector"
    )

    print("Training complete!")
    print("Best model saved to: runs/detect/dice_detector/weights/best.pt")