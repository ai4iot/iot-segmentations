from src.models.classification.model_builder import ModelBuilder  # Importing ModelBuilder from the sibling module
from src.tools import DataPreparation, ModelUtils  # Importing DataPreparation and ModelUtils from the parent module
import torch
import logging
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # Importing tqdm for progress bars
from io import StringIO
import wandb


class Trainer:
    """
    Trainer class for training and validating a neural network model.

    Args:
    - data_preparation (DataPreparation): An instance of the DataPreparation class for loading and preparing data.
    - model (ModelBuilder): An instance of the ModelBuilder class for building the neural network model.
    - device (str): Device to run the model on ('cuda' if available, else 'cpu').
    - criterion (torch.nn.Module): Loss function used for training the model (default: nn.CrossEntropyLoss()).
    - learning_rate (float): Learning rate for the optimizer (default: 0.001).
    - epochs (int): Number of training epochs (default: 10).
    - output_dir (str): Output directory for saving trained models and plots (default: '../../runs').

    Methods:
    - _train(trainloader): Private method for training the model on the training data.
    - _validate(testloader): Private method for validating the model on the test data.
    - run(): Main method for running the training and validation process.

    """

    def __init__(self, data_preparation: DataPreparation, model: ModelBuilder,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 criterion=nn.CrossEntropyLoss(), learning_rate=0.001, epochs=10, output_dir='../../runs'):
        """
        Initialize the Trainer class with specified parameters.

        """

        self.data_preparation = data_preparation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_builder = model
        self.optimizer = optim.Adam(self.model_builder.model.parameters(), lr=self.learning_rate)
        # TODO: Add support for other optimizers.
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir

    def _train(self, trainloader):
        """
        Train the model on the training data.

        Args:
        - trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.

        Returns:
        - Tuple: Tuple containing the epoch loss and accuracy.

        """
        tqdm_output = StringIO()
        self.model_builder.model.train()
        logging.info('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            counter += 1
            image, labels = data
            image = image.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            # Forward pass.
            outputs = self.model_builder.model(image)
            # Calculate the loss.
            loss = self.criterion(outputs, labels)
            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            # Backpropagation
            loss.backward()
            # Update the weights.
            self.optimizer.step()

        # Loss and accuracy for the complete epoch.
        logging.info(tqdm_output.getvalue())
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
        wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc})

        return epoch_loss, epoch_acc

    def _validate(self, testloader):
        """
        Validate the model on the test data.

        Args:
        - testloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
        - Tuple: Tuple containing the epoch loss and accuracy.

        """
        self.model_builder.model.eval()
        logging.info('Validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                counter += 1

                image, labels = data
                image = image.to(self.device)
                labels = labels.to(self.device)
                # Forward pass.
                outputs = self.model_builder.model(image)
                # Calculate the loss.
                loss = self.criterion(outputs, labels)
                valid_running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()

        # Loss and accuracy for the complete epoch.
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
        wandb.log({"valid_loss": epoch_loss, "valid_acc": epoch_acc})
        return epoch_loss, epoch_acc

    def run(self):
        """
        Run the training and validation process.

        """
        # Load the training and validation data.
        print(self.model_builder.device)
        self.model_builder.model.to(self.device)
        dataset_train, dataset_valid, dataset_classes = self.data_preparation.get_datasets()
        logging.info('Dataset loaded')
        logging.info('Number of training images: {}'.format(len(dataset_train)))
        logging.info('Number of validation images: {}'.format(len(dataset_valid)))
        logging.info('Number of classes: {}'.format(len(dataset_classes)))

        # Create the data loaders.
        train_loader, valid_loader = self.data_preparation.get_data_loaders(dataset_train, dataset_valid)
        logging.info('Data loaders created.')
        logging.info('Training parameters:')
        logging.info('Learning rate: {}'.format(self.learning_rate))
        logging.info('Epochs: {}'.format(self.epochs))
        logging.info('Model: {}'.format(self.model_builder.model_name))
        logging.info('Criterion: {}'.format(self.criterion))
        logging.info('Device: {}'.format(self.device))
        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in self.model_builder.model.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model_builder.model.parameters() if p.requires_grad
        )
        logging.info(f'{total_trainable_params:,} training parameters.\n')

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        # Start the training and validation.
        for epoch in range(self.epochs):
            logging.info(f"Epoch {epoch + 1} of {self.epochs}")
            train_epoch_loss, train_epoch_acc = self._train(train_loader)
            valid_epoch_loss, valid_epoch_acc = self._validate(valid_loader)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            logging.info(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            logging.info(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            logging.info('-' * 50)

        # Save the trained model.

        direc = ModelUtils.create_new_pre_dir(base_dir=self.output_dir)
        ModelUtils.save_model(epochs=self.epochs, model=self.model_builder.model, optimizer=self.optimizer,
                              criterion=self.criterion, pretrained=self.data_preparation.pretrained,
                              model_name=self.model_builder.model_name, directory=direc)
        ModelUtils.save_plots(train_acc=train_acc, valid_acc=valid_acc, train_loss=train_loss, valid_loss=valid_loss,
                              pretrained=self.data_preparation.pretrained, model_name=self.model_builder.model_name,
                              directory=direc)
        wandb.finish()
