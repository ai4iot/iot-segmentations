import imutils
import numpy as np
import torch
import cv2
from model import build_model
from torchvision import transforms
import argparse
import logging

IMAGE_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Class names.
class_names = ['non_person', 'person']

# -c/--camera: Port of the camera you want to use.
# -s/--size: Size of the window to show the images.
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-c', '--camera', type=int, default=0,
                             help='Port the camera is connected to (0-99).'
                                  'Default is 0. If you only have one camera,'
                                  'you should always use 0.')
argument_parser.add_argument('-s', '--size', type=int, default=600, help='Width of the window where we will see the '
                                                                         'stream'
                                                                         'of the camera. (Maintains the relationship of'
                                                                         'aspect with respect to width)')
argument_parser.add_argument(
    '-m', '--model-name', type=str, default='efficientnet_b0',
    dest='model-name', help='Model to use for training: efficientnet_b0, resnet18'
)

argument_parser.add_argument(
    '-w', '--weights', type=str, default='../weights/model_pretrained_True_prueba2.pth',
    dest='weights', help='Model weights to use for testing.'
)

arguments = vars(argument_parser.parse_args())

model = build_model(pretrained=False, fine_tune=False, num_classes=len(class_names),
                    model_name=arguments['model-name'])
checkpoint = torch.load(arguments['weights'], map_location=DEVICE)
logging.info(f"Loaded model weights from {arguments['weights']}")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

camera = cv2.VideoCapture(arguments['camera'])

while True:
    grabbed, frame = camera.read()

    if not grabbed:
        break
    # Preprocess the image
    transformed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    transformed_frame = transform(transformed_frame)
    transformed_frame = torch.unsqueeze(transformed_frame, 0)
    transformed_frame = transformed_frame.to(DEVICE)

    # Forward pass through the image.
    outputs = model(transformed_frame)
    outputs = outputs.detach().cpu().numpy()
    pred_class_name = class_names[np.argmax(outputs[0])]
    print(outputs[0])

    color = (0, 255, 0) if pred_class_name == 'person' else (0, 0, 255)

    # Annotate the image with prediction.
    cv2.putText(
        frame, f"Pred: {pred_class_name}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, color, 2, lineType=cv2.LINE_AA
    )
    cv2.imshow('Yo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
