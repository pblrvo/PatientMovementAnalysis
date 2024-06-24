import keras
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from src.hyperparameter_tuning import MyHyperModel
from utils.model_visualization import plot_training_history
import sys
sys.path.append('./utils')

def model_training(train_data, train_labels, validation_data, validation_labels, fold_no):
    filepath = "results/models/video_classifier_model.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

    tuner = kt.RandomSearch(
            MyHyperModel(),
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory='results',
            project_name=f'hyperparam_tuning_fold_{fold_no}',
            overwrite=True,
        )

    tuner.search(train_data, train_labels, epochs=200, validation_data=(validation_data, validation_labels), callbacks=[checkpoint, early_stopping])
    print(f"Hyperparameter tuning completed for fold {fold_no}")
    print(f"Retrieving the best model for fold {fold_no}...")
    best_model = tuner.get_best_models(num_models=1)[0]
    print(f"Starting training of the best model for fold {fold_no}...")
    history = best_model.fit(
        train_data, train_labels,
        validation_data=(validation_data, validation_labels),
        epochs=200,
        callbacks=[early_stopping],
        verbose=1
    )
    # Collect true and predicted labels for confusion matrix
    validation_prediction = np.argmax(best_model.predict(validation_data), axis=1)
    
    best_model.save("results/models/video_classifier_model.keras")
    print(f"Model saved to {"results/models/video_classifier_model.keras"}")
    
    val_accuracy = max(history.history['val_accuracy'])

    plot_training_history(history)
    return best_model, val_accuracy, validation_prediction


def model_evaluation(model, test_data, test_labels):

    model = keras.models.load_model("results/models/video_classifier_model.keras")
    _, accuracy = model.evaluate(test_data, test_labels)

    return accuracy