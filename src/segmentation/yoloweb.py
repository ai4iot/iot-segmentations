import cv2
from ultralytics import YOLO
from flask import Flask
from flask import render_template
from flask import Response

app = Flask(__name__)


def generate():
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()

        results = model(frame)

        frame = results[0].plot()

        frame = cv2.resize(frame, (960, 540))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    return 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    import sys

    model = YOLO("yolov8s.pt")

    app.run(host='0.0.0.0', port=5000, debug=False)