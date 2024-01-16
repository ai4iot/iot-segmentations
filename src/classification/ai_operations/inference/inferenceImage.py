import torch
import cv2
import numpy as np
import glob as glob
import os
from ..models import ModelBuilder
import argparse
import logging


class LocalInfence():

    def __init__(self, model_builder: ModelBuilder,
                 output_dir='../../runs',
                 input_dir='../../input/test/esp-camera',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Metrics object.

        Args:
        - model_builder (ModelBuilder): Instance of ModelBuilder.
        - output_dir (str): Output directory for saving results.
        - input_dir (str): Input directory containing test images.
        - device (str): Device to use for inference ('cuda' or 'cpu').

        """

        self.model_builder = model_builder
        self.device = device
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.class_names = sorted(
            [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))])



















logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Class names.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model-name', type=str, default='efficientnet_b0',
    dest='model-name', help='Model to use for training: efficientnet_b0, resnet18'
)
parser.add_argument(
    '-w', '--weights', type=str, default='../../weights/person_nonperson_efficientnet.pth',
    dest='weights', help='Model weights to use for testing.'
)

parser.add_argument('-i', '--input', type=str, default='../../input/test/esp-camera',
                    dest='input', help='Path to the input folder.'
                    )

parser.add_argument('-o', '--output', type=str, default='predictions',
                    dest='output', help='Path to the output folder where images predictions are save.'
                    )

parser.add_argument('-s', '--save', type=bool, default=False,
                    dest='save', help='True for save the labeled images into the output folder, false '
                                      'for not save them but visualize them.'
                    )

args = vars(parser.parse_args())



class_names = [name for name in os.listdir(args['input']) if os.path.isdir(os.path.join(args['input'], name))]

print(class_names)
# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=len(class_names), model_name=args['model-name'])
checkpoint = torch.load(args['weights'], map_location=DEVICE)
logging.info('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
logging.info('Model loaded.')
print(model.eval())
model.eval()

# Get all the test image paths.
all_image_paths = glob.glob(f"{args['input']}/*/*.jpeg")
new_dir = create_new_pre_dir(args['output'])
# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    gt_class_name = image_path.split(os.path.sep)[-2]
    # Read the image and create a copy.
    try:
        image = cv2.imread(image_path)
        orig_image = image.copy()
        if image is None:
            logging.error(f"Error reading image {image_path}")
            continue
        image_name = os.path.basename(image_path)
        image = image_normalization(image)
        image = image.to(DEVICE)

        outputs = model(image)
        outputs = outputs.detach().numpy()
        pred_class_name = class_names[np.argmin(outputs[0])]

        #           print(outputs[0])
        #print(f"GT: {gt_class_name}, Pred: {pred_class_name.lower()}")
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
        print(class_names.index(pred_class_name.lower()))
        if args['save']:
            cv2.imwrite(f"{new_dir}/{image_name}", orig_image)
        else:
            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(0)
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
    continue
