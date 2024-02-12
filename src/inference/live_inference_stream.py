import numpy as np

from src.models import ModelBuilder
from src.tools import ModelUtils

import sys
import cv2
from flask import Response
from flask import Flask
from flask import render_template
import torch

arguments_str = sys.argv[1]
arguments_list = arguments_str.split("@")
class_names = arguments_list[1].split("-")

model = ModelBuilder(
    name=arguments_list[2],
    pretrained=bool(arguments_list[3].lower()),
    fine_tune=bool(arguments_list[4].lower()),
    num_classes=int(arguments_list[5]),
    model_name=arguments_list[6],
    weights=arguments_list[7],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



print(device)
app = Flask(__name__)


def generate():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        model.model.eval()
        transformed_frame = ModelUtils.image_normalization(frame, 224)
        model.model.to(device)
        transformed_frame= transformed_frame.to(device)
        outputs = model.model(transformed_frame)

        outputs = outputs.cpu().detach().numpy()
        pred_class_name = class_names[np.argmax(outputs[0])]
        color = (0, 255, 0) if pred_class_name == 'person' else (0, 0, 255)
        cv2.putText(
            frame, f"Pred: {pred_class_name}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, color, 2, lineType=cv2.LINE_AA
        )

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


app.run(host='0.0.0.0', port=5000, debug=False)