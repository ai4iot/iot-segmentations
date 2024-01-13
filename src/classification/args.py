import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model-name',
        type=str,
        default='efficientnet_b0',
        dest='model_name',
        help='Model to use for training: efficientnet_b0, resnet18'
    )

    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='../../weights/model_pretrained_True_prueba2.pth',
        dest='weights',
        help='Model weights to use for testing.'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default='../../input/test/esp-camera',
        dest='input',
        help='Path to the input folder.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='predictions',
        dest='output',
        help='Path to the output folder where images predictions are saved.'
    )

    parser.add_argument(
        '-s', '--save',
        type=bool,
        default=False,
        dest='save',
        help='True to save the labeled images into the output folder, false '
             'to not save them but visualize them.'
    )

    parser.add_argument(
        '-c', '--camera',
        type=int,
        default=0,
        help='Port the camera is connected to (0-99).'
             'Default is 0. If you only have one camera,'
             'you should always use 0.'
    )

    parser.add_argument(
        '-v', '--visualize',
        type=int,
        default=600,
        help='Width of the window where we will see the '
             'stream'
             'of the camera. (Maintains the relationship of'
             'aspect with respect to width)'
    )

    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=10,
        help='Number of epochs to train our network for'
    )

    parser.add_argument(
        '-pt', '--pretrained',
        dest='pretrained',
        action='store_false',
        default=True,
        help='Whether to use pretrained weights or not'
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        type=float,
        dest='learning_rate',
        default=0.0001,
        help='Learning rate for training the model'
    )

    parser.add_argument(
        '-', '--mode',
        type=str,
        default='train',
        dest='mode',
        help='Mode to run the script in: train, test'
    )

    parser.add_argument(
        '-fn', '--fine-tune',
        dest='fine_tune',
        action='store_true',
        default=False,
        help='Whether to fine-tune all layers or not.'
    )

    return parser.parse_args()
