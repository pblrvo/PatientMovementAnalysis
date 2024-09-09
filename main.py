from src.data_collection import load_csv
from utils.download_data import download_file_from_drive, decompress_file
from utils.model_visualization import plot_confusion_matrix
from src.data_preparation import build_tensors, build_tensors_for_test
from src.model_compilation import model_training, model_evaluation
from sklearn.model_selection import KFold
import numpy as np

if __name__ == '__main__':
    #download_file_from_drive(file_id='1rqwyCpqf82_zhyD9Jay4m7Y8HXJmW4pJ', output='./resources/labeled_keypoints.csv.zip')
    #decompress_file(zip_file='./resources/labeled_keypoints.csv.zip', output_folder='./resources', password='')
    videos_dataframe = load_csv(filepath='./resources/labeled_keypoints.csv')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    val_accuracies = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    #Used later to plot Confusion Matrix
    all_labels_true = []
    all_labels_pred = []


    for train_and_val_index, test_index in kf.split(videos_dataframe):
        print(f"Training fold {fold_no}...")
        train_and_val_dataframe, test_dataframe = videos_dataframe.iloc[train_and_val_index], videos_dataframe.iloc[test_index]

        train_X, train_Y, val_X, val_Y, classes = build_tensors(train_and_val_dataframe)

        print(f"X_train shape: {train_X.shape}, y_train shape: {train_Y.shape}")
        print(f"X_val shape: {val_X.shape}, y_val shape: {val_Y.shape}")

        print(f"Starting hyperparameter tuning for fold {fold_no}...")

        best_model, fold_accuracy, validation_prediction = model_training(train_X, train_Y, val_X, val_Y, fold_no)

        test_X, test_Y = build_tensors_for_test(test_dataframe)

        print(f"Fold {fold_no} validation accuracy: {fold_accuracy:.4f}")
        val_accuracies.append(fold_accuracy)
        all_labels_true.extend(val_Y)
        all_labels_pred.extend(validation_prediction)
        print(f"Training completed for fold {fold_no}")

        accuracy, precision, recall, f1 = model_evaluation(best_model, test_X, test_Y)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test precision: {round(precision * 100, 2)}%")
        print(f"Test recall: {round(recall * 100, 2)}%")
        print(f"Test F1 score: {round(f1 * 100, 2)}%")

        # Append metrics for each fold
        test_accuracies.append(accuracy)
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_f1s.append(f1)

        fold_no +=1


    
    print(f"Average validation accuracy over {kf.n_splits} folds: {np.mean(val_accuracies):.4f}")
    print(f"Average test accuracy over {kf.n_splits} folds: {np.mean(test_accuracies):.4f}")
    print(f"Average test precision over {kf.n_splits} folds: {np.mean(test_precisions):.4f}")
    print(f"Average test recall over {kf.n_splits} folds: {np.mean(test_recalls):.4f}")
    print(f"Average test F1 score over {kf.n_splits} folds: {np.mean(test_f1s):.4f}")


    plot_confusion_matrix(all_labels_true, all_labels_pred, classes)