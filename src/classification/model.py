import torchvision.models as models
import torch.nn as nn
import torch
import logging


def build_model(pretrained=True, fine_tune=True, num_classes=10, model_name='efficientnet_b0'):
    if pretrained:
        logging.info('Loading pre-trained weights')
    else:
        logging.info('Not loading pre-trained weights')

    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
        logging.info('EfficientNet-B0 loaded.')
    elif model_name == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=num_classes)
        logging.info('ResNet-18 loaded.')
    else:
        logging.info(f'Model {model_name} not implemented.')
        raise NotImplementedError(f'Model {model_name} not implemented.')

    # Freeze all the layers.
    if fine_tune:
        logging.info('Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        logging.info('Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    return model
