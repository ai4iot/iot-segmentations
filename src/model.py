import torchvision.models as models
import torch.nn as nn
import torch


def build_model(pretrained=True, fine_tune=True, num_classes=10, model_name='efficientnet_b0'):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')

    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
        print('[INFO]: EfficientNet-B0 loaded.')
    elif model_name == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        model.fc = nn.Linear(in_features=512, out_features=num_classes)
        print('[INFO]: ResNet-18 loaded.')
    else:
        print('[INFO]: Model not implemented.')
        raise NotImplementedError(f'Model {model_name} not implemented.')

    # Freeze all the layers.
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    return model
