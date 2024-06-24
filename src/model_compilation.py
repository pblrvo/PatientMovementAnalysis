import keras
from data_preparation import build_tensors
from data_collection import load_csv
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from hyperparameter_tuning import MyHyperModel
import sys
sys.path.append('./utils')
from model_visualization import plot_confusion_matrix, plot_training_history
from sklearn.model_selection import KFold, train_test_split

def tune_hyperparameters(X, y):
    print("Hyperparameter tuning starting")
    filepath = "/tmp/video_classifier_model.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    tuner = kt.RandomSearch(
        MyHyperModel(),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='hyperparam_tuning',
        overwrite=True,
    )
    tuner.search(X_train, y_train, epochs=200, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])
    print("Hyperparameter tuning completed")
    print("Retrieving the best model")
    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model

def cross_validate_model(best_model, train_data, train_labels, test_data, test_labels, classes):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    val_accuracies = []
    all_y_true = []
    all_y_pred = []

    for train_index, val_index in kf.split(train_data):
        print(f"Training fold {fold_no}...")
        X_train, X_val = train_data[train_index], train_data[val_index]
        y_train, y_val = train_labels[train_index], train_labels[val_index]

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        history = best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            verbose=1
        )
        print(f"Training completed for fold {fold_no}")

        val_accuracy = max(history.history['val_accuracy'])
        val_accuracies.append(val_accuracy)
        print(f"Fold {fold_no} validation accuracy: {val_accuracy:.4f}")
        
        # Collect true and predicted labels for confusion matrix
        y_pred = np.argmax(best_model.predict(X_val), axis=1)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        fold_no += 1

    print(f"Average validation accuracy over {kf.n_splits} folds: {np.mean(val_accuracies):.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(all_y_true, all_y_pred, classes=classes)

    # Plot training history
    plot_training_history(history)

    # Save the final model
    best_model.save("/tmp/video_classifier_model.keras")
    print("Final model saved")

    best_model = keras.models.load_model("/tmp/video_classifier_model.keras")
    
    _, accuracy = best_model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return best_model

def run_experiment():
    videos_df = load_csv("./resources/labeled_keypoints.csv")
    train_data, train_labels, test_data, test_labels, classes = build_tensors(videos_df)
    best_model = tune_hyperparameters(train_data, train_labels)
    cross_validate_model(best_model, train_data, train_labels, test_data, test_labels, classes)
    
model = run_experiment()