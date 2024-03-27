# Happy-Sad Image Classifier: Deep Learning for Emotion Recognition

This project implements a deep learning model to classify human facial expressions as happy or sad in images. It leverages the TensorFlow and Keras libraries to build and train a convolutional neural network (CNN) model.

<p align="center">
<img src="https://github.com/simranwarraich/happy-sad-face-classifier/blob/main/happy-face-sad-face.jpg" width="550" height="400" valign="center">
</p>

## Getting Started
The project requires the following Python libraries:

- tensorflow
- tensorflow-gpu (recommended for faster training)
- opencv-python
- matplotlib

## The script performs the following steps:
1. Installs dependencies
2. Preprocesses the image data (removing invalid images, resizing)
3. Loads the image data as a TensorFlow dataset
4. Splits the data into training, validation, and testing sets
5. Builds a convolutional neural network model
6. Trains the model on the training data
7. Evaluates the model's performance on the validation and test sets
8. Saves the trained model for future use
9. Optionally tests the model on a single image
  
## Additional Notes:

The script utilizes TensorBoard for visualization of the training process. To view the logs, run tensorboard --logdir=logs in your terminal after running the script (assuming the log directory is named "logs").

## Project Structure:
The project consists of the following files:

1. [**image-classification-using-tensorflow.ipynb:**](https://github.com/simranwarraich/happy-sad-face-classifier/blob/main/image-classification-using-tensorflow.ipynb) Contains the main script for data processing, model building, training, and evaluation.
2. **README.md:** This file (you are reading it now!) provides an overview of the project.
3. **Dataset:** The dataset used for this project is available at [Kaggle Dataset](https://www.kaggle.com/datasets/simrankaur21/happy-sad-emotions-image-data)


## Model Performance
The performance of the model is evaluated using metrics like accuracy, precision, and recall. You can find these metrics reported in the script's output.

## Saving the Model
The trained model is saved as happysadimageclassifier.h5 in the models directory within the project. You can load this model for future use in other applications.

## Who,When,Why?
**Author**: Simranjeet Kaur<br>
**Version**: 1

