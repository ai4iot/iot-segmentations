import time

from flask import Flask
from flask import render_template
from flask import Response
import numpy as np
import torch
import cv2
from jinja2.compiler import generate
from ..models import ModelBuilder
from ..tools import ModelUtils


app = Flask(__name__)


class WebInference():

    def __init__(self, model_builder: ModelBuilder,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 class_names=['non_person', 'person'],
                 data_source=0,
                 image_size=224):
        """
        Initialize Metrics object.

        Args:
        - model_builder (ModelBuilder): Instance of ModelBuilder.
        - device (str): Device to use for inference ('cuda' or 'cpu').
        - class_names (list): List of class names.
        - data_source (int): Source of the image data
        - IMAGE_SIZE (int): Size of the image to be processed

        """

        self.model_builder = model_builder
        self.device = device
        self.class_names = class_names
        self.data_source = data_source
        self.IMAGE_SIZE = image_size
        self.data = cv2.VideoCapture(self.data_source)


    def init_stream(self):
        """
        Initialize the stream of the data source

        Returns:
        - None

        """
        # to device
        self.model_builder.model.to(self.device)
        self.model_builder.model.eval()

        while True:

            grabbed, frame = self.data.read()
            if not grabbed:
                break
            # Preprocess the image
            frame = ModelUtils.image_normalization(frame, self.IMAGE_SIZE)
            transformed_frame = transformed_frame.to(self.device)

            # Forward pass through the image.
            # starting_time = time.time()
            # outputs = model(transformed_frame)
            # logging.info(f'Inference time: {(time.time() - starting_time) * 1000} microseconds''')
            outputs = outputs.detach().cpu().numpy()
            pred_class_name = self.class_names[np.argmax(outputs[0])]
            color = (0, 255, 0) if pred_class_name == 'person' else (0, 0, 255)
            # Annotate the image with prediction.
            cv2.putText(
                frame, f"Pred: {pred_class_name}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, lineType=cv2.LINE_AA
            )
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')

    @app.route("/")
    def index(self):
        return render_template("templates/index.html")

    @app.route("/video_feed")
    def video_feed(self):
        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
