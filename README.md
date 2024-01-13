# Person Detection Project

## Overview

This project aims to explore and compare different models for person detection.
The current implementations include EfficientNet_B0 and ResNet18. MobileNetV2 is also included for comparison purposes, 
but at the moment it is not an option in the parameters. You can use it in 
[person-detection-esp32s3](https://github.com/curso-verano-iot-uah/person-detection-esp32s3)
The goal is to evaluate the performance of these models in terms of accuracy, speed, and resource requirements.
In the future, more models will be added to further expand the analysis and options.  
  

* [Models results and comparison](#models-results-and-comparison)
* [Future Models](#future-models)
* [Dataset](#dataset)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Live inference](#live-inference)

## Models results and comparison


### MobileNetV2
- **Accuracy**  95.1%
- **F1 Score:** 0.96
- **Confusion Matrix:**

| Actual/Predicted | person | nonperson |
|------------------|--------|-----------|
| person           | 564    | 36        |
| nonperson        | 12     | 388       |


### EfficientNet
- **Accuracy** 95.8%
- **F1 Score:** 0.96
- **Confusion Matrix:**

| Actual/Predicted | person | nonperson |
|------------------|--------|-----------|
| person           | 597    | 3         |
| nonperson        | 39     | 361       |

### ResNet18
- **Accuracy** 97.1%
- **F1 Score:** 0.96
- **Confusion Matrix:**

| Actual/Predicted | person | nonperson |
|------------------|--------|-----------|
| person           | 571    | 29        |
| nonperson        | 0      | 400       |


## Future Models

In the future, additional models will be incorporated into the project to provide a broader 
range of options for person detection. Contributions and suggestions for new models are welcome.

## Dataset

For training, we have used the [person_dataset](input/person_dataset). Contains
two classes (person and nonperson). It is made up with images from coco dataset and
images taken grom an ESP32-CAM.  
For testing we have used the [esp-camera](input/test/esp-camera). Contains 1000 images all taken 
from an ESP32-CAM.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/curso-verano-iot-uah/iot-segmentations.git
   cd iot-segmentations/src
   ```
2. Create a conda environment:

   ```bash
   conda env create -f environment.yaml
   ```
   
## Usage

### Training

For training, you can use the following command:
```bash
  python trainer.py -m <model_name> -e <epochs> -pt <self> -lr <learning_rate> -n <name>
```  

- **model_name**: Name of the model to use. Currently, the options are: `efficientnet`and `resnet18`.
- **epochs**: Number of epochs to train the model.
- **self**: Use a self model. Options: `True` or `False`.
- **learning_rate**: Learning rate to use in the training process.
- **name**: Name of the model to save the weights and the results.

All the results will be saved in this path `~/Documents/training_results/<name>`.

### Testing

You have to options for testing the models:

1. Visualize each image with the predictions:      

    ```bash
    python inferenceImage.py -m <model_name> -w <weights_path>
    ```
2. Obtain all the metrics for the test dataset:

    ```bash
    python confusion_matrix.py -m <model_name> -w <weights_path>
    ```
   
You will also a get a plot with the confusion matrix.
   
## Live inference
You can use your webcam to make live inference with the models you've trained. We have two options:

### Local inference

With [live-inference-local.py](src/live-inference-local.py). You can use the following command:

```bash
python live-inference-local.py -c <camera_port> -s <window_size> -m <model_name> -w <weights_path>
```

### Web streaming inference

With [live-inference-web.py](src/live-inference-stream.py). You can use the following command:

```bash
python live_inference_stream.py -c <camera_port> -s <window_size> -m <model_name> -w <weights_path>
```
Then you can access to the streaming in your browser with this url: `http://<ip>:5000/`






