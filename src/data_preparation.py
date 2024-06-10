from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from data_collection import load_features

videos_df = load_features('./resources/JSON/')


#Split data into train & test dataframes while ensuring label representation
def split_data_with_label_representation(df):
    labels = df["label"].to_numpy()

  # StratifiedShuffleSplit for label-aware splitting
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

  # Splitting based on label groups
    for train_index, test_index in sss.split(df, labels):
        train_X = df.iloc[train_index, :-1]  
        train_y = labels[train_index]
        test_X = df.iloc[test_index, :-1]
        test_y = labels[test_index]


    return train_X, test_X, train_y, test_y

def build_tensors(df):
    
  train_X, test_X, train_y, test_y = split_data_with_label_representation(df)

  train_X_tensor = []
  for video in train_X['keypoints']:
    # Iterate through frames in the video
    frame_tensors = []
    for frame in video:
      # Convert each frame into a tensor
      frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
      frame_tensors.append(frame_tensor)
    # Append the list of frame tensors (representing the video) to the final list
    train_X_tensor.append(frame_tensors)

  test_X_tensor = []
  for video in test_X['keypoints']:
    frame_tensors = []
    for frame in video:
      frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
      frame_tensors.append(frame_tensor)
    test_X_tensor.append(frame_tensors)

  return train_X_tensor, train_y, test_X_tensor, test_y