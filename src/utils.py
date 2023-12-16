import torch
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.style.use('ggplot')




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
