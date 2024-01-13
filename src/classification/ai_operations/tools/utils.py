import torch
import cv2
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import os
import logging

matplotlib.style.use('ggplot')


def image_normalization(image, IMAGE_SIZE=224):
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

    return image


def save_model(epochs, model, optimizer, criterion, pretrained, name, model_name):
    """
    Function to save the trained model to disk.
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


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, name, model_name):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    userdir = os.path.expanduser('~')
    directory = f"{userdir}/Documents/training_results/{name}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{directory}/accuracy_{model_name}_pretrained_{pretrained}_{name}.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{directory}/loss_{model_name}_pretrained_{pretrained}_{name}.png")

def obtain_dir_number(base_dir):

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

def create_new_pre_dir(base_dir):

    next_number = obtain_dir_number(base_dir)

    new_dir = f"{base_dir}/pre_{next_number}"
    os.makedirs(new_dir)

    return new_dir


