import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class DataPreparation:
    """
    Class to prepare the datasets.
    """

    def __init__(self, root_dir='../../../../input/person_dataset', valid_split=0.1, image_size=224, batch_size=16,
                 num_workers=6, pretrained=True):
        """
        Constructor for DataPreparation class.

        Args:
        - root_dir: Root directory of the dataset.
        - valid_split: Percentage of data to be used for validation.
        - image_size: Size of the images after resizing.
        - batch_size: Batch size for DataLoader.
        - num_workers: Number of parallel processes for data preparation.
        - pretrained: Boolean, indicating whether to use pretrained weights.
        """
        self.root_dir = root_dir
        self.valid_split = valid_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pretrained = pretrained

    def get_train_transform(self):
        """
        Returns a set of training transformations.

        Returns:
        - train_transform: Set of transformations for training data.
        """
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ToTensor(),
            self.normalize_transform()
        ])
        return train_transform

    def get_valid_transform(self):
        """
        Returns a set of validation transformations.

        Returns:
        - valid_transform: Set of transformations for validation data.
        """
        valid_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            self.normalize_transform()
        ])
        return valid_transform

    def normalize_transform(self):
        """
        Returns normalization transformations based on the 'pretrained' flag.

        Returns:
        - normalize: Normalization transformations.
        """
        if self.pretrained:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        return normalize

    def get_datasets(self):
        """
        Function to prepare the datasets.

        Returns:
        - dataset_train: Training dataset.
        - dataset_valid: Validation dataset.
        - Dataset.classes: Class names.
        """
        dataset = datasets.ImageFolder(
            self.root_dir,
            transform=self.get_train_transform()
        )
        dataset_test = datasets.ImageFolder(
            self.root_dir,
            transform=self.get_valid_transform()
        )
        dataset_size = len(dataset)
        valid_size = int(self.valid_split * dataset_size)
        indices = torch.randperm(len(dataset)).tolist()
        dataset_train = Subset(dataset, indices[:-valid_size])
        dataset_valid = Subset(dataset_test, indices[-valid_size:])
        return dataset_train, dataset_valid, dataset.classes

    def get_data_loaders(self, dataset_train, dataset_valid):
        """
        Prepares the training and validation data loaders.

        Args:
        - dataset_train: The training dataset.
        - dataset_valid: The validation dataset.

        Returns:
        - train_loader: DataLoader for training data.
        - valid_loader: DataLoader for validation data.
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
