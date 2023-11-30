import torch
import cv2
import numpy as np
import glob as glob
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import build_model
from torchvision import transforms

# Constants.
DATA_PATH = '../input/test'
IMAGE_SIZE = 224
DEVICE = 'cpu'
class_names = ['non_person', 'person']

# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=2)
checkpoint = torch.load('../weights/model_pretrained_True_prueba4A.pt', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print('Loading trained model weights...')

# Lists to store ground truth and predicted labels.
true_labels = []
predicted_labels = []

# Get all the test image paths.
all_image_paths = glob.glob(f"{DATA_PATH}/*/*.jpg")

# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    gt_class_name = image_path.split(os.path.sep)[-2]
    true_labels.append(class_names.index(gt_class_name))

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error al leer la imagen: {image_path}")
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
        print(f"Error al procesar la imagen {image_path}: {str(e)}")
    continue

conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
