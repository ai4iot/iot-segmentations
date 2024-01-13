from ..tools import DataPreparation,ModelUtils
import torch
import logging
import torch.optim as optim
import torch.nn as nn
import tqdm.auto as tqdm


class Trainer:

    def __init__(self, data_preparation, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 criterion=nn.CrossEntropyLoss, learning_rate=0.001, epochs=10):
        self.data_preparation = data_preparation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = criterion
        self.device = device
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _train(self, trainloader):
        self.model.train()
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
            outputs = self.model(image)
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
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
        return epoch_loss, epoch_acc

    def _validate(self, testloader):
        self.model.eval()
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
                outputs = self.model(image)
                # Calculate the loss.
                loss = self.criterion(outputs, labels)
                valid_running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()

        # Loss and accuracy for the complete epoch.
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
        return epoch_loss, epoch_acc

    def run(self):
        #Load the training and validation data.

        dataset_train, dataset_valid, dataset_classes = self.data_preparation.get_datasets()
        logging.info('Dataset laoded')
        logging.info('Number of training images: {}'.format(len(dataset_train)))
        logging.info('Number of validation images: {}'.format(len(dataset_valid)))
        logging.info('Number of classes: {}'.format(len(dataset_classes)))

        # Create the data loaders.
        train_loader, valid_loader = self.data_preparation.get_data_loaders(dataset_train, dataset_valid)
        logging.info('Data loaders created.')
        logging.info('Training parameters:')
        logging.info('Learning rate: {}'.format(self.learning_rate))
        logging.info('Epochs: {}'.format(self.epochs))
        logging.info('Model: {}'.format(self.model))
        logging.info('Criterion: {}'.format(self.criterion))
        logging.info('Device: {}'.format(self.device))

        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logging.info(f'{total_trainable_params:,} training parameters.\n')

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        #Start the training and validation.
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

        direc = ModelUtils.create_new_pre_dir(base_dir="~/models")
        ModelUtils.save_model(epochs=self.epochs, model=self.model, optimizer=self.optimizer,
                              criterion=self.criterion, pretrained=self.data_preparation.pretrained,
                              model_name=self.model.model_name, directory=direc)
        ModelUtils.save_plots(train_acc=train_acc, valid_acc=valid_acc, train_loss=train_loss, valid_loss=valid_loss,
                              pretrained=self.data_preparation.pretrained, model_name=self.model.model_name,
                              directory=direc)




















