# Patient Movement Analysis for Parkinson's Disease Severity Classification

## Overview
This project aims to develop a deep learning model that can classify the severity of Parkinson's Disease (PD) in patients based on their movement patterns captured in videos. The model utilizes a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to process temporal sequences of body keypoints extracted from videos. It predicts the severity level of PD into one of four classes: NORMAL, MILD, MODERATE, and SEVERE.

## Project Structure

- **resources/**: This is where the data will be stored
  
- **src/**:
  - `data_preprocessing.py`: Contains functions for preprocessing the keypoints data.
  - `model_compilation.py`: Contains functions for defining, compiling, and training the deep learning model.
  - `hyperparameter_tuning.py`: Defines the hyperparameter search space and the model architecture to be tuned using Keras Tuner.
  - `utils.py`: (Optional) Contains any additional utility functions.
  - `main.py`: The main script to run the project. It handles data loading, preprocessing, training, evaluation, and potentially saving the trained model.

## Dependencies
- Python 3.7+

## Installation
```bash
pip install -r requirements.txt

## Model Architecture

The deep learning model employs a hybrid architecture:

### Convolutional Layers
- Multiple 1D convolutional layers (`Conv1D`) with increasing filter sizes are applied to the input sequence of keypoints to extract spatial features within each frame.
- Batch normalization (`BatchNormalization`) layers are added after each convolutional layer to stabilize training and improve generalization.

### LSTM Layers
- Two LSTM layers process the output of the convolutional layers to capture temporal dependencies in the movement patterns across frames.
- Dropout layers are added after each LSTM to prevent overfitting.

### Dense Layers
- Several fully connected (`Dense`) layers further process the features learned by the convolutional and LSTM layers.
- Dropout is applied again to enhance generalization.

### Output Layer
- A final dense layer with a softmax activation function produces the probability distribution over the four PD severity classes.

### Citations
- The data used in this project is the data obtained in the research paper: Pose-Based Gait Analysis for Diagnosis of Parkinson’s Disease
  - Tee Connie, Timilehin B. Aderinola, Thian Song Ong, Michael Goh, Bayu Erfianto, Bedy Purnama, “Pose-based Gait Analysis for Diagnosis of Parkinson’s Disease”, Algorithms, 15(12), 474, 2022. DOI: https://doi.org/10.3390/a15120474.

## Disclaimer
This project is for research and educational purposes only and should not be used for making actual medical diagnoses or treatment decisions. Always consult with a qualified healthcare professional for any health concerns.
