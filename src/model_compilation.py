from keras import models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime
import keras_tuner as kt
import numpy as np
from src.hyperparameter_tuning import MyHyperModel
from src.balance_data_generator import BalancedDataGenerator
import tensorflow as tf
from utils.model_visualization import plot_training_history
import sys
sys.path.append('./utils')
from typing import Tuple, Dict, Any

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights to handle imbalanced datasets.

    Args:
        y (np.ndarray): Array of labels.

    Returns:
        Dict[int, float]: Dictionary of class weights.
    """
    class_counts = np.bincount(y)
    total_samples = len(y)
    n_classes = len(class_counts)
    weights = total_samples / (n_classes * class_counts)
    return dict(enumerate(weights))

def model_training(train_data: np.ndarray, train_labels: np.ndarray, validation_data: np.ndarray, validation_labels: np.ndarray, fold_no: int) -> Tuple[models.Model, float, np.ndarray]:
    """
    Train the model with hyperparameter tuning and balanced data generator.

    Args:
        train_data (np.ndarray): Training data.
        train_labels (np.ndarray): Training labels.
        validation_data (np.ndarray): Validation data.
        validation_labels (np.ndarray): Validation labels.
        fold_no (int): Fold number for cross-validation.

    Returns:
        Tuple[models.Model, float, np.ndarray]: Trained model, validation accuracy, and validation predictions.
    """
    log_dir = "results/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    filepath = "results/models/video_classifier_model.keras"
    checkpoint = ModelCheckpoint(
        filepath, save_best_only=True, verbose=1
    )
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

    class_weights = compute_class_weights(train_labels)
    train_generator = BalancedDataGenerator(train_data, train_labels, batch_size=16)
    tuner = kt.RandomSearch(
        MyHyperModel(),
        objective='val_accuracy',
        executions_per_trial=1,
        max_trials=100,
        directory='results',
        project_name=f'hyperparam_tuning_fold_{fold_no}',
        overwrite=True,
    )

    tuner.search(train_generator, epochs=200, class_weight=class_weights, batch_size=16,  validation_data=(validation_data, validation_labels), callbacks=[checkpoint, early_stopping, tensorboard_callback])
    print(f"Hyperparameter tuning completed for fold {fold_no}")
    print(f"Retrieving the best model for fold {fold_no}...")
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Starting training of the best model for fold {fold_no}...")

    history = best_model.fit(
        train_data, train_labels,
        class_weight=class_weights,
        validation_data=(validation_data, validation_labels),
        batch_size=16,
        epochs=200,
        callbacks=[early_stopping, tensorboard_callback],
        verbose=1
    )

    tensorboard_callback.set_model(best_model)

    best_model.save(filepath)
    print(f"Model saved to {filepath}")
    
    val_accuracy = max(history.history['val_accuracy'])
    
    validation_prediction = np.argmax(best_model.predict(validation_data), axis=1)

    plot_training_history(history, fold_no)
    return best_model, val_accuracy, validation_prediction

def model_evaluation(model: models.Model, test_data: np.ndarray, test_labels: np.ndarray) -> float:
    """
    Evaluate the model on test data.

    Args:
        model (models.Model): Trained model.
        test_data (np.ndarray): Test data.
        test_labels (np.ndarray): Test labels.

    Returns:
        float: Test accuracy.
    """
    model = models.load_model("results/models/video_classifier_model.keras")
    _, accuracy = model.evaluate(test_data, test_labels)
    return accuracy