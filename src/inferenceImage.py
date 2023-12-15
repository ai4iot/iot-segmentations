import torch
import cv2
import numpy as np
import glob as glob
import os
from model import build_model
from torchvision import transforms
import argparse
# Constants.
DATA_PATH = '../input/test/esp-camera'
IMAGE_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Class names.
class_names = ['non_person', 'person']

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model-name', type=str, default='efficientnet_b0',
    dest='model-name', help='Model to use for training: efficientnet_b0, resnet18'
)
parser.add_argument(
    '-w', '--weights', type=str, default='model_pretrained_True_prueba2.pth',
    dest='weights', help='Model weights to use for testing.'
)

args = vars(parser.parse_args())


# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=len(class_names), model_name=args['model-name'])
checkpoint = torch.load(args['weights'], map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
print(model.eval())
model.eval()

# Get all the test image paths.
all_image_paths = glob.glob(f"{DATA_PATH}/*/*.jpeg")
# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    gt_class_name = image_path.split(os.path.sep)[-2]
    # Read the image and create a copy.
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)

    # Forward pass through the image.
    outputs = model(image)
    outputs = outputs.detach().cpu().numpy()
    pred_class_name = class_names[np.argmax(outputs[0])]
    print(outputs[0])
    print(f"GT: {gt_class_name}, Pred: {pred_class_name.lower()}")
    # Annotate the image with ground truth.
    cv2.putText(
        orig_image, f"GT: {gt_class_name}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
    )
    # Annotate the image with prediction.
    cv2.putText(
        orig_image, f"Pred: {pred_class_name.lower()}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
    )
    cv2.imshow('Result', orig_image)
    cv2.waitKey(0)
    cv2.imwrite(f"../outputs/exp1/{gt_class_name}.png", orig_image)