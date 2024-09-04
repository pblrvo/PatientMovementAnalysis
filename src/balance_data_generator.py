import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
from typing import Tuple, List

class BalancedDataGenerator(Sequence):
    """
    A data generator for balanced batches of data.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize the data generator.

        Args:
            x (np.ndarray): Input data.
            y (np.ndarray): Labels.
            batch_size (int, optional): Size of the batches. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data at the end of each epoch. Defaults to True.
        """
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.classes: np.ndarray = np.unique(y)
        self.num_classes: int = len(self.classes)
        self.indices: List[np.ndarray] = [np.where(y == c)[0] for c in self.classes]
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Number of batches per epoch.

        Returns:
            int: Number of batches per epoch.
        """
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.

        Args:
            idx (int): Index of the batch.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Batch of data and labels.
        """
        batch_x: List[np.ndarray] = []
        batch_y: List[np.ndarray] = []
        for class_idx in range(self.num_classes):
            indices: np.ndarray = self.indices[class_idx]
            selected_indices: np.ndarray = indices[idx * (self.batch_size // self.num_classes):
                                                    (idx + 1) * (self.batch_size // self.num_classes)]
            batch_x.extend(self.x[selected_indices])
            batch_y.extend(self.y[selected_indices])
        
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self) -> None:
        """
        Shuffle the data at the end of each epoch.
        """
        if self.shuffle:
            for idx in range(self.num_classes):
                np.random.shuffle(self.indices[idx])