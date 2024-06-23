import keras
from data_preparation import build_tensors
from data_collection import load_csv
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt
from hyperparameter_tuning import MyHyperModel, EPOCHS
from model_visualization import plot_confusion_matrix, plot_training_history
from sklearn.model_selection import KFold, train_test_split

def tune_hyperparameters(X, y):
    print("Hyperparameter tuning starting")
    filepath = "/tmp/video_classifier_model.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)  # 20% validation set

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
    tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])
    print("Hyperparameter tuning completed")
    print("Retrieving the best model")
    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model

def cross_validate_model(best_model, data, labels, classes):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    val_accuracies = []
    all_y_true = []
    all_y_pred = []

    for train_index, val_index in kf.split(data):
        print(f"Training fold {fold_no}...")
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        history = best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
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

    return best_model

def run_experiment():
    videos_df = load_csv("./resources/labeled_keypoints.csv")
    X, y, classes = build_tensors(videos_df)
    best_model = tune_hyperparameters(X, y)
    cross_validate_model(best_model, X, y, classes)
    
model = run_experiment()