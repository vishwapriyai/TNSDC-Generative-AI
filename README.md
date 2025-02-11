# TNSDC Generative AI  

# Traffic sign recognition using CNN


## Overview

This project implements a Traffic Sign Recognition system using Convolutional Neural Networks (CNN) with Keras and TensorFlow. The model is trained to classify traffic signs into 43 different classes based on images provided in the dataset.

## Table of Contents

- [Installation](#installation)
- [Data Exploration](#data-exploration)
- [Model Architecture](#model-architecture)
- [Training and Validation](#training-and-validation)
- [Evaluation Metrics](#evaluation-metrics)
- [GUI Application](#gui-application)


## Installation

To set up the environment, install the required libraries using the following command:

```bash
# Install required libraries
%pip install tensorflow keras sklearn matplotlib pandas pillow
```

## Data Exploration

The dataset is divided into training and testing data. The training data consists of images grouped into 43 classes, each representing a type of traffic signal.

## Model Architecture

The CNN model is built using Keras and consists of the following layers:

1. **Convolutional Layers**:
   - Two layers with 32 filters and a kernel size of (5,5).
   - Two layers with 64 filters and a kernel size of (3,3).

2. **Pooling Layers**:
   - MaxPooling layers to reduce dimensionality.

3. **Dropout Layers**:
   - To prevent overfitting.

4. **Dense Layers**:
   - A fully connected layer with 256 units followed by a dropout layer.
   - An output layer with 43 units (one for each class) using softmax activation.

## Training and Validation

The model is trained for 15 epochs with a batch size of 32. The training and validation accuracy and loss are monitored.

```python
# Model compilation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
anc = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

```
## Evaluation Metrics

After training, the model is evaluated using various metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## GUI Application
- A simple GUI application is created using Tkinter to allow users to upload images and classify them. The application displays the predicted traffic sign based on the uploaded image.
![Output Image](TNSDC-Generative-AI/Output/1.png)
![Output Image](TNSDC-Generative-AI/Output/2.png)
