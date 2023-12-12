# MobileNetV2 vs EfficientNet

## Introduction
We are going to compare the performance of two image classification models, MobileNetV2 and EfficientNet.
The goal is to identify which model performs better on this dataset and why.  
It must be taken into account that MobileNetV2 is designed for embedded systems and microcontrollers.
On the other hand, EfficientNet is much heavier. Also it is important to mention that
to use MobileNetV2 we are going to use [Edge Impulse](https://www.edgeimpulse.com/) and for EfficientNet
we have made a fine-tuning of the model using this model [EfficientNet](https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/).

## Dataset
The dataset consist of 1000 images of 2 classes [person](../input/test/esp-camera/person) and [non_person](../input/test/esp-camera/non_person).
All images have been taken form the ESP32-CAM camera, with a resolution of 320x240 pixels.  
Th MobileNetV2 down-samples the images to 160x160 pixels. 


## Evaluation Metrics
For the metrics we are going to use the following:
- Precision: TP / (TP + FP)
- F1 Score.
- Confusion Matrix.


## Results

### MobileNetV2
- **Accuracy**  95.1%
- **F1 Score:** 0.96
- **Confusion Matrix:**

| Actual/Predicted | person | nonperson |
|-------------------|--------|-----------|
| person            | 564    | 36        |
| nonperson         | 12     | 388       |


### EfficientNet
- **Accuracy** 95.8%
- **F1 Score:** 0.96
- **Confusion Matrix:**

| Actual/Predicted | person | nonperson |
|-------------------|--------|-----------|
| person            | 597    | 3         |
| nonperson         | 39     | 361       |




