import torchvision.models as models
import torch.nn as nn
import torch
import logging


class ModelBuilder:
    def __init__(self, nombre):
        self.model = None
        self.model_name = nombre

    def build_model(self, pretrained=True, fine_tune=True, num_classes=10, model_name='efficientnet_b0'):
        """
        Builds and configures the model.

        Args:
        - pretrained: Boolean, indicating whether to use pretrained weights.
        - fine_tune: Boolean, indicating whether to fine-tune all layers.
        - num_classes: Number of output classes.
        - model_name: Name of the model to build.

        Returns:
        - model: The constructed and configured model.
        """
        if pretrained:
            logging.info('Loading pre-trained weights')
        else:
            logging.info('Not loading pre-trained weights')

        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
            logging.info('EfficientNet-B0 loaded.')
        elif model_name == 'resnet18':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=num_classes)
            logging.info('ResNet-18 loaded.')
        else:
            logging.info(f'Model {model_name} not implemented.')
            raise NotImplementedError(f'Model {model_name} not implemented.')

        # Freeze or fine-tune layers.
        self._set_requires_grad(fine_tune)

        return self.model

    def _set_requires_grad(self, fine_tune):
        """
        Helper function to set requires_grad attribute for model parameters.

        Args:
        - fine_tune: Boolean, indicating whether to fine-tune all layers.
        """
        if fine_tune:
            logging.info('Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        else:
            logging.info('Freezing hidden layers...')
            for params in self.model.parameters():
                params.requires_grad = False
