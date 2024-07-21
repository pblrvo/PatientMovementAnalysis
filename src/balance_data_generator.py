import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

class BalancedDataGenerator(Sequence):
    def __init__(self, x, y, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.indices = [np.where(y == c)[0] for c in self.classes]
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for class_idx in range(self.num_classes):
            indices = self.indices[class_idx]
            selected_indices = indices[idx * (self.batch_size // self.num_classes):
                                       (idx + 1) * (self.batch_size // self.num_classes)]
            batch_x.extend(self.x[selected_indices])
            batch_y.extend(self.y[selected_indices])
        
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            for idx in range(self.num_classes):
                np.random.shuffle(self.indices[idx])