import keras
from data_preparation import build_tensors
import numpy as np
import data_collection
from model_visualization import plot_confusion_matrix, plot_training_history
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt
from hyperparameter_tuning import MyHyperModel, EPOCHS
from sklearn.model_selection import KFold

videos_df = data_collection.load_features('./resources/JSON/')
X, y = build_tensors(videos_df)

def run_experiment():
    filepath = "/tmp/video_classifier_model.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    val_accuracies = []

    for train_index, val_index in kf.split(X):
        print(f"Training fold {fold_no}...")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        print(f"Starting hyperparameter tuning for fold {fold_no}...")
        tuner = kt.RandomSearch(
            MyHyperModel(),
            objective='val_accuracy',
            max_trials=10,  # Reduce trials for testing
            executions_per_trial=1,
            directory='my_dir',
            project_name=f'hyperparam_tuning_fold_{fold_no}',
            overwrite=True,
        )

        tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), batch_size=8, callbacks=[checkpoint, early_stopping])
        print(f"Hyperparameter tuning completed for fold {fold_no}")

        print(f"Retrieving the best model for fold {fold_no}...")
        best_model = tuner.get_best_models(num_models=1)[0]

        print(f"Starting training of the best model for fold {fold_no}...")
        history = best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=8,
            callbacks=[early_stopping],
            verbose=1
        )
        print(f"Training completed for fold {fold_no}")

        val_accuracy = max(history.history['val_accuracy'])
        val_accuracies.append(val_accuracy)
        print(f"Fold {fold_no} validation accuracy: {val_accuracy:.4f}")
        
        fold_no += 1

    print(f"Average validation accuracy over {kf.n_splits} folds: {np.mean(val_accuracies):.4f}")

    print("Starting final training on all data...")
    tuner = kt.RandomSearch(
        MyHyperModel(),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='hyperparam_tuning_final',
        overwrite=True,
    )

    tuner.search(X, y_encoded, epochs=EPOCHS, validation_split=0.15, batch_size=8, callbacks=[checkpoint, early_stopping])
    print("Hyperparameter tuning completed for final model")

    print("Retrieving the best model for final training...")
    best_model = tuner.get_best_models(num_models=1)[0]

    print("Starting training of the best final model...")
    history = best_model.fit(
        X, y_encoded,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=1
    )
    print("Training of the best final model completed")

    plot_training_history(history)
    best_model.save(filepath)
    print(f"Model saved to {filepath}")
    
    best_model = keras.models.load_model(filepath)
    
    _, accuracy = best_model.evaluate(X, y_encoded)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return best_model

model = run_experiment()
