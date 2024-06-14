import keras
from data_preparation import build_tensors
import data_collection
from model_visualization import plot_confusion_matrix, plot_training_history
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt
from hyperparameter_tuning import MyHyperModel, EPOCHS, MAX_SEQ_LENGTH

videos_df = data_collection.load_features('./resources/JSON/')
train_data, train_labels, test_data, test_labels = build_tensors(videos_df)

train_data_padded = pad_sequences(train_data, maxlen=MAX_SEQ_LENGTH, padding='post')
test_data_padded = pad_sequences(test_data, maxlen=MAX_SEQ_LENGTH, padding='post')

def run_experiment():
    filepath = "/tmp/video_classifier_model.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_best_only=True, verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
    
    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels)
    test_labels_encoded = le.transform(test_labels)

    tuner = kt.RandomSearch(
        MyHyperModel(),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='hyperparam_tuning',
        overwrite=True,
    )

    tuner.search(train_data_padded, train_labels_encoded, epochs=EPOCHS, validation_split=0.15, callbacks=[checkpoint, early_stopping])
    best_model = tuner.get_best_models(num_models=1)[0]
    history = best_model.fit(
        train_data_padded, train_labels_encoded,
        validation_split=0.15,
        epochs=EPOCHS,
    )
    plot_training_history(history)
    best_model.save(filepath)
    
    best_model = keras.models.load_model(filepath)
    
    _, accuracy = best_model.evaluate(test_data_padded, test_labels_encoded)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return best_model

model = run_experiment()