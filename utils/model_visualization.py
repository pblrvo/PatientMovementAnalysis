import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Any, List

def plot_training_history(history: Any, fold_no: int) -> None:
    """
    Plot training and validation accuracy and loss values.

    Args:
        history (Any): Training history object.
        fold_no (int): Fold number for cross-validation.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(f'results/visualization/training_history_fold{fold_no}.png')

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], classes: List[str]) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        classes (List[str]): List of class names.
    """
    cm: Any = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('results/visualization/confusion_matrix.png')