import cv2
from ultralytics import YOLO


def main(args):
    # cap = cv2.VideoCapture('rtsp://admin:GramLaboratori0@192.168.79.120:554/h264Preview_01_sub')  # IP Camera
    cap = cv2.VideoCapture(0)  # WebCam

    while (True):
        ret, frame = cap.read()

        results = model(frame)

        frame = results[0].plot()

        frame = cv2.resize(frame, (960, 540))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # click q to stop capturing
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    import sys

    model = YOLO("yolo-Weights/yolov8n.pt")

    sys.exit(main(sys.argv))