import os

import cv2
from ultralytics import YOLO
from flask import Flask
from flask import render_template
from flask import Response
import paho.mqtt.client as mqtt

app = Flask(__name__)


def generate():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("rtsp://admin:reolink2023@192.168.79.108:554/h264Preview_01_sub")
    while (True):
        ret, frame = cap.read()
        results = model.predict(frame, stream=True)

        # Process the detected objects
        for r in results:
            i = 0
            for c, box in zip(r.boxes.cls, r.boxes.xyxy.cpu().numpy().astype(int)):
                xmin, ymin, xmax, ymax = box

                # Draw bounding box
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)
                cv2.putText(img=frame, text=f"{names[int(c)]}", org=(xmin, ymin - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(0, 255, 0), thickness=2)

                if names[int(c)] == 'person':
                    print("Alerta")
                    message_payload = f'cam1: Person detected - {xmin}, {ymin}, {xmax}, {ymax}'
                    client.publish(mqtt_topic, message_payload)

                i = i + 1

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
    return render_template("../classification/ai_operations/inference/templates/index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    import sys

    model = YOLO("yolov8s.pt")
    print(os.getcwd())
    names = model.names
    mqtt_broker = "localhost"
    mqtt_port = 1883
    mqtt_topic = "/alerta"
    client = mqtt.Client()
    client.connect(mqtt_broker, mqtt_port, 60)

    app.run(host='0.0.0.0', port=5002, debug=False)