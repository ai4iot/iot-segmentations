import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import logging


class ModelUtils:
    def __init__(self):
        # Setting the matplotlib style
        plt.style.use('ggplot')

    @staticmethod
    def image_normalization(image, image_size=224):
        """
        Normalize and prepare an image for model input.

        Args:
        - image: Input image (BGR format).
        - IMAGE_SIZE: Size to resize the image.

        Returns:
        - Normalized and processed image.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        return image

    @staticmethod
    def save_model(epochs, model, optimizer, criterion, pretrained, name, model_name):
        """
        Save the trained model to disk.

        Args:
        - epochs: Number of training epochs.
        - model: The trained model.
        - optimizer: The optimizer used during training.
        - criterion: The loss criterion.
        - pretrained: Boolean indicating whether the model is pretrained.
        - name: Name associated with the saved model.
        - model_name: Name of the model.

        Saves the model checkpoint to the specified directory.
        """
        userdir = os.path.expanduser('~')
        directory = f"{userdir}/Documents/training_results/{name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, f"{directory}/{model_name}_pretrained_{pretrained}_{name}.pt")

    @staticmethod
    def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, name, model_name):
        """
        Save loss and accuracy plots to disk.

        Args:
        - train_acc: List of training accuracies over epochs.
        - valid_acc: List of validation accuracies over epochs.
        - train_loss: List of training losses over epochs.
        - valid_loss: List of validation losses over epochs.
        - pretrained: Boolean indicating whether the model is pretrained.
        - name: Name associated with the saved plots.
        - model_name: Name of the model.

        Saves the accuracy and loss plots to the specified directory.
        """
        userdir = os.path.expanduser('~')
        directory = f"{userdir}/Documents/training_results/{name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Accuracy plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_acc, color='green', linestyle='-',
            label='train accuracy'
        )
        plt.plot(
            valid_acc, color='blue', linestyle='-',
            label='validation accuracy'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f"{directory}/accuracy_{model_name}_pretrained_{pretrained}_{name}.png")

        # Loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-',
            label='train loss'
        )
        plt.plot(
            valid_loss, color='red', linestyle='-',
            label='validation loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{directory}/loss_{model_name}_pretrained_{pretrained}_{name}.png")

    def obtain_dir_number(base_dir):
        """
        Obtain the next available directory number.

        Args:
        - base_dir: Base directory path.

        Returns:
        - The next available directory number.
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            logging.info(f"Created directory {base_dir}")

        try:
            dirs_names = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
            numbers = [int(name.split('_')[1]) for name in dirs_names]
            next_number = 1 if not numbers else max(numbers) + 1
        except Exception as e:
            print(e)
            next_number = 1

        return next_number

    @staticmethod
    def create_new_pre_dir(self, base_dir):
        """
        Create a new directory with a unique 'pre_' prefix.

        Args:
        - base_dir: Base directory path.

        Returns:
        - The path of the newly created directory.
        """
        next_number = self.obtain_dir_number(base_dir)
        new_dir = f"{base_dir}/pre_{next_number}"
        os.makedirs(new_dir)
        return new_dir
