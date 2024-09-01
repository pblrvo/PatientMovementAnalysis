from keras import models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ModelCheckpoint
import datetime
import keras_tuner as kt
import numpy as np
from src.hyperparameter_tuning import MyHyperModel
from src.balance_data_generator import BalancedDataGenerator
import tensorflow as tf
from utils.model_visualization import plot_training_history
import sys
sys.path.append('./utils')

def compute_class_weights(y):
    class_counts = np.bincount(y)
    total_samples = len(y)
    n_classes = len(class_counts)
    weights = total_samples / (n_classes * class_counts)
    return dict(enumerate(weights))

def model_training(train_data, train_labels, validation_data, validation_labels, fold_no):

    # TensorBoard callback
    log_dir = "results/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Model checkpoint callback
    filepath = "results/models/video_classifier_model.keras"
    checkpoint = ModelCheckpoint(
        filepath, save_best_only=True, verbose=1
    )
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

    #class_weights = compute_class_weights(train_labels)
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

    tuner.search(train_generator, epochs=200, batch_size=16,  validation_data=(validation_data, validation_labels), callbacks=[checkpoint, early_stopping, tensorboard_callback])
    print(f"Hyperparameter tuning completed for fold {fold_no}")
    print(f"Retrieving the best model for fold {fold_no}...")
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Starting training of the best model for fold {fold_no}...")

    history = best_model.fit(
        train_data, train_labels,
        validation_data=(validation_data, validation_labels),
        batch_size=16,
        epochs=200,
        callbacks=[early_stopping, tensorboard_callback],
        verbose=1
    )

    # Log metrics using TensorBoard
    tensorboard_callback.set_model(best_model)

    best_model.save(filepath)
    print(f"Model saved to {filepath}")
    
    val_accuracy = max(history.history['val_accuracy'])
    # Collect true and predicted labels for confusion matrix
    validation_prediction = np.argmax(best_model.predict(validation_data), axis=1)

    plot_training_history(history, fold_no)
    return best_model, val_accuracy, validation_prediction

def model_evaluation(model, test_data, test_labels):
    model = models.load_model("results/models/video_classifier_model.keras")
    _, accuracy = model.evaluate(test_data, test_labels)
    return accuracy