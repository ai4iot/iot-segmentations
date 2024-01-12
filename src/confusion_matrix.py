import torch
import cv2
import numpy as np
import glob as glob
import os
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from model import build_model
from torchvision import transforms
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


# Constants.
DATA_PATH = '../input/test/esp-camera'
IMAGE_SIZE = 224
DEVICE = 'cpu'
class_names = ['non_person', 'person']

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model-name', type=str, default='efficientnet_b0',
    dest='model-name', help='Model to use for training: efficientnet_b0, resnet18'
)
parser.add_argument(
    '-w', '--weights', type=str, default='../weights/model_pretrained_True_prueba2.pth',
    dest='weights', help='Model weights to use for testing.'
)
args = vars(parser.parse_args())

# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=len(class_names), model_name=args['model-name'])
logging.info('Loading trained model weights...')
checkpoint = torch.load(args['weights'], map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
logging.info('Model loaded successfully.')
print(model.eval())
model.eval()

model.eval()

# Lists to store ground truth and predicted labels.
true_labels = []
predicted_labels = []

# Get all the test image paths.
all_image_paths = glob.glob(f"{DATA_PATH}/*/*.jpeg")

# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    gt_class_name = image_path.split(os.path.sep)[-2]
    true_labels.append(class_names.index(gt_class_name))

    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Error reading image {image_path}")
            continue

        orig_image = image.copy()

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

        outputs = model(image)
        outputs = outputs.detach().numpy()
        pred_class_name = class_names[np.argmax(outputs[0])]
        predicted_labels.append(class_names.index(pred_class_name.lower()))
        # print('GT: ' + gt_class_name + ' Pred: ' + pred_class_name)

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
    continue

conf_matrix = confusion_matrix(true_labels, predicted_labels)
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


