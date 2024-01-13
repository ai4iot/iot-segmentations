import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class DataPreparation:
    """
    Class to prepare the datasets.
    """

    def __init__(self, root_dir='../../../../input/person_dataset', valid_split=0.1, image_size=224, batch_size=16,
                 num_workers=6, pretrained=True):
        self.root_dir = root_dir
        self.valid_split = valid_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pretrained = pretrained

    def get_train_transform(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ToTensor(),
            self.normalize_transform(self.pretrained)
        ])
        return train_transform

    # Validation transforms
    def get_valid_transform(self):
        valid_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            self.normalize_transform(self.pretrained)
        ])
        return valid_transform

    # Image normalization transforms.
    def normalize_transform(self):
        if self.pretrained:  # Normalization for pre-trained weights.
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:  # Normalization when training from scratch.
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        return normalize

    def get_datasets(self):
        """
        Function to prepare the Datasets.
        :param pretrained: Boolean, True or False.
        Returns the training and validation datasets along
        with the class names.
        """
        dataset = datasets.ImageFolder(
            self.root_dir,
            transform=(self.get_train_transform(self.image_size, self.pretrained))
        )
        dataset_test = datasets.ImageFolder(
            self.root_dir,
            transform=(self.get_valid_transform(self.image_size, self.pretrained))
        )
        dataset_size = len(dataset)
        # Calculate the validation dataset size.
        valid_size = int(self.valid_split * dataset_size)
        # Randomize the data indices.
        indices = torch.randperm(len(dataset)).tolist()
        # Training and validation sets.
        dataset_train = Subset(dataset, indices[:-valid_size])
        dataset_valid = Subset(dataset_test, indices[-valid_size:])
        return dataset_train, dataset_valid, dataset.classes

    def get_data_loaders(self, dataset_train, dataset_valid):
        """
        Prepares the training and validation data loaders.
        :param dataset_train: The training dataset.
        :param dataset_valid: The validation dataset.
        Returns the training and validation data loaders.
        """
        train_loader = DataLoader(
            dataset_train, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )
        valid_loader = DataLoader(
            dataset_valid, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )
        return train_loader, valid_loader
