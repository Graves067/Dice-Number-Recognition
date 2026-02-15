import cv2 as cv
import easyocr
import os
from ultralytics import YOLO 


if __name__ == '__main__':

    model = YOLO(r"runs/detect/dice_detector9/weights/best.pt")

    camera = cv.VideoCapture(0)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Failed to grab Frame")
            break


        results = model(frame, conf = 0.6, verbose = False)

        for result in results:
            for box in result.boxes:

                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                confidence = float(box.conf)
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                cv.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2)
                label = f"{cls_name.upper()}  {confidence:.0%}"
                cv.putText(frame, label, (x1, y1 - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                # Draw center dot
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

       
        cv.imshow("Dice Detector", frame)

        # Quit on Q key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()