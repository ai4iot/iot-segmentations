import matplotlib.pyplot as plt
import torch
import cv2
import glob as glob
import os
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from ..models import ModelBuilder
from ..tools import ModelUtils
import logging

matplotlib.use('Agg')


class Metrics:

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
        self.true_labels = []
        self.predicted_labels = []
        self.class_names = sorted(
            [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))])

    def _obtain_images_paths(self):
        """
        Obtain the paths of all the images in the input directory.

        Returns:
        - List: List containing the paths of all the images in the input directory.

        """
        return glob.glob(f"{self.input_dir}/*/*.jpeg")

    def obtain_metrics(self):

        """
        Obtain and print confusion matrix, precision, recall, and F1 score metrics.

        This function processes all images in the input directory, predicts their labels using the trained model,
        and calculates performance metrics.

        Returns:
        - None

        """

        all_image_paths = self._obtain_images_paths()
        self.model_builder.model_name = self.model_builder.model.to(self.device)
        self.model_builder.model.eval()

        logging.info(f'Starting inference with {self.device}')
        logging.info(f'Found {len(all_image_paths)} images in {self.input_dir}')
        print(f'Found {len(all_image_paths)} images in {self.input_dir}')

        for image_path in all_image_paths:
            gt_class_name = image_path.split(os.path.sep)[-2]
            try:
                # Read and normalize the image.

                image = cv2.imread(image_path)
                if image is None:
                    logging.error(f"Error reading image {image_path}")
                    continue
                self.true_labels.append(self.class_names.index(gt_class_name))
                image = ModelUtils.image_normalization(image)
                image = image.to(self.device)

                # Perform inference.
                outputs = self.model_builder.model(image)
                outputs = torch.argmax(outputs, dim=1)
                outputs = outputs.cpu().detach().numpy()
                pred_class_name = self.class_names[outputs[0]]
                self.predicted_labels.append(self.class_names.index(pred_class_name.lower()))

            except Exception as e:
                logging.error(f"Error processing image {image_path}: {str(e)}")
            continue

        # Calculate and display confusion matrix.
        print("len true labels", len(self.true_labels))
        print("len predicted labels", len(self.predicted_labels))

        conf_matrix = confusion_matrix(self.true_labels, self.predicted_labels)
        print(conf_matrix)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=self.class_names)
        disp.plot(cmap='Greens', values_format='d')
        #plt.show()

        new_dir = ModelUtils.create_new_pre_dir(self.output_dir)
        logging.info(f'Saving confusion matrix to {new_dir}/confusion_matrix.png')
        plt.savefig(f"{new_dir}/confusion_matrix.png")

        # Calculate and print precision, recall, and F1 score.
        precision = precision_score(self.true_labels, self.predicted_labels)
        recall = recall_score(self.true_labels, self.predicted_labels)
        f1 = f1_score(self.true_labels, self.predicted_labels)

        print('Precision: ' + str(precision))
        logging.info(f'Precision: {precision}')
        print('Recall: ' + str(recall))
        logging.info(f'Recall: {recall}')
        print('F1: ' + str(f1))
        logging.info(f'F1: {f1}')