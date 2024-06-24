from src.data_collection import load_csv
from utils.download_data import download_file_from_drive, decompress_file
from utils.model_visualization import plot_confusion_matrix
from src.data_preparation import build_tensors
from src.model_compilation import model_training
from sklearn.model_selection import KFold
import numpy as np

if __name__ == '__main__':
    download_file_from_drive(file_id='1rqwyCpqf82_zhyD9Jay4m7Y8HXJmW4pJ', output='./resources/labeled_keypoints.csv.zip')
    decompress_file(zip_file='./resources/labeled_keypoints.csv.zip', output_folder='./resources', password='')
    videos_dataframe = load_csv(filepath='./resources/labeled_keypoints.csv')
    train_data, train_labels, test_data, test_labels, classes = build_tensors(videos_dataframe=videos_dataframe)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    val_accuracies = []
    #Used later to plot Confusion Matrix
    all_labels_true = []
    all_labels_pred = []

    for train_index, val_index in kf.split(train_data):
        print(f"Training fold {fold_no}...")
        X_train, X_val = train_data[train_index], train_data[val_index]
        Y_train, Y_val = train_labels[train_index], train_labels[val_index]

        print(f"X_train shape: {X_train.shape}, y_train shape: {Y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {Y_val.shape}")

        print(f"Starting hyperparameter tuning for fold {fold_no}...")

        fold_accuracy, validation_prediction = model_training(X_train, Y_train, X_val, Y_val, fold_no)
        print(f"Fold {fold_no} validation accuracy: {fold_accuracy:.4f}")
        val_accuracies.append(fold_accuracy)
        all_labels_true.extend(Y_val)
        all_labels_pred.extend(validation_prediction)
        print(f"Training completed for fold {fold_no}")
        fold_no +=1


    print(f"Average validation accuracy over {kf.n_splits} folds: {np.mean(val_accuracies):.4f}")


    plot_confusion_matrix(all_labels_true, all_labels_pred, classes)