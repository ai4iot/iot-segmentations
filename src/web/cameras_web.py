import time
import cv2
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import re

app = Flask(__name__)
cameras = []

# Regular expression pattern to match camera URL
regex = r'\d'

mqtt_messages_per_second = 2
mqtt_message_timer = 0


def publish_mqtt_message(camera_url, xmin, ymin, xmax, ymax):
    global mqtt_message_timer
    global mqtt_messages_per_second
    global mqtt_topic
    global client

    current_time = time.time()
    time_elapsed = current_time - mqtt_message_timer
    time_interval = 1 / mqtt_messages_per_second

    if time_elapsed >= time_interval:
        message_payload = f'{camera_url.split("@")[-1].split(":")[0]}: Person detected - {xmin}, {ymin}, {xmax}, {ymax}'
        client.publish(mqtt_topic, message_payload)
        mqtt_message_timer = current_time


@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html', cameras=cameras, mqtt_messages_per_second=mqtt_messages_per_second)


@app.route('/add_camera', methods=['POST'])
def add_camera():
    """Add a camera to the list."""
    if re.fullmatch(regex, request.form['camera_url']):
        camera_url = int(request.form['camera_url'])
        print(camera_url)
    else:
        camera_url = request.form['camera_url']
    cameras.append(camera_url)
    return render_template('index.html', cameras=cameras, mqtt_messages_per_second=mqtt_messages_per_second)


@app.route('/remove_camera/<int:camera_index>', methods=['GET'])
def remove_camera(camera_index):
    """Remove a camera from the list."""
    if 0 <= camera_index < len(cameras):
        del cameras[camera_index]
    return render_template('index.html', cameras=cameras, mqtt_messages_per_second=mqtt_messages_per_second)


def generate_frames(camera_url):
    """Generate frames from the camera feed."""
    cap = cv2.VideoCapture(camera_url)
    while True:
        ret, frame = cap.read()
        results = model.predict(frame, stream=True)

        # Process the detected objects
        for r in results:
            i = 0
            for c, box in zip(r.boxes.cls, r.boxes.xyxy.cpu().numpy().astype(int)):
                xmin, ymin, xmax, ymax = box

                # Draw bounding box
                width = xmax - xmin
                height = ymax - ymin

                font_scale = min(width, height) / 100

                # Draw the bounding box
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

                font_thickness = max(1, int(font_scale))
                font_scale = max(0.5, font_scale)

                cv2.putText(img=frame, text=f"{names[int(c)]}", org=(xmin, ymin - int(10 * font_scale)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 255, 0),
                            thickness=font_thickness)

                if names[int(c)] == 'person':
                    print("Alert")
                    # Publish MQTT message if a person is detected
                    publish_mqtt_message(camera_url, xmin, ymin, xmax, ymax)

                i = i + 1

        # Encode frame as JPEG image
        # TODO: Añadir flag para qe se guarden las imágenes en un directorio
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    """Stream video feed from the specified camera."""
    if 0 <= camera_index < len(cameras):
        return Response(generate_frames(cameras[camera_index]), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid camera index"


@app.route('/update_mqtt', methods=['POST'])
def update_mqtt():
    """Update the MQTT messages per second parameter."""
    global mqtt_messages_per_second
    if 'mqtt_messages_per_second' in request.form:
        mqtt_messages_per_second = int(request.form['mqtt_messages_per_second'])
    return render_template('index.html', mqtt_messages_per_second=mqtt_messages_per_second, cameras=cameras)


if __name__ == '__main__':
    # Load YOLO model and names
    model = YOLO("yolov8n.pt")
    names = model.names

    # Attempt to connect to MQTT broker
    try:
        mqtt_broker = "localhost"
        mqtt_port = 1883
        mqtt_topic = "/alerta"
        client = mqtt.Client()
        client.connect(mqtt_broker, mqtt_port, 60)
    except:
        print("Failed to connect to MQTT broker")

    # Start Flask application
    app.run(debug=True)
