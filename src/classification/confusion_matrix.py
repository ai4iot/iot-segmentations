import torch
import cv2
import numpy as np
import glob as glob
import os
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from model import build_model
from utils import image_normalization
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


# Constants.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model-name', type=str, default='efficientnet_b0',
    dest='model-name', help='Model to use for training: efficientnet_b0, resnet18'
)
parser.add_argument(
    '-w', '--weights', type=str, default='../../weights/model_pretrained_True_prueba2.pth',
    dest='weights', help='Model weights to use for testing.'
)

parser.add_argument('-i', '--imput', type=str, default='../../input/test/esp-camera',
                    dest='input', help='Path to the input folder.'
                    )

args = vars(parser.parse_args())

class_names = [name for name in os.listdir(args['input']) if os.path.isdir(os.path.join(args['input'], name))]

print(class_names)



# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=len(class_names), model_name=args['model-name'])
logging.info('Loading trained model weights...')

checkpoint = torch.load(args['weights'], map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
logging.info('Model loaded successfully.')

print(model.eval())
model.eval()

# Lists to store ground truth and predicted labels.
true_labels = []
predicted_labels = []

# Get all the test image paths.
all_image_paths = glob.glob(f"{args['input']}/*/*.jpeg")

# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    gt_class_name = image_path.split(os.path.sep)[-2]
    true_labels.append(class_names.index(gt_class_name))

    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Error reading image {image_path}")
            continue

        image = image_normalization(image)
        image = image.to(DEVICE)

        outputs = model(image)
        outputs = outputs.detach().numpy()
        pred_class_name = class_names[np.argmax(outputs[0])]
        predicted_labels.append(class_names.index(pred_class_name.lower()))


    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
    continue

conf_matrix = confusion_matrix(true_labels, predicted_labels)
print(true_labels)
print(predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['non_person', 'person'])
disp.plot(cmap='Greens', values_format='d')
matplotlib.pyplot.show()

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('F1: ' + str(f1))


