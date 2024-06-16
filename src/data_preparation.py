from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from data_collection import load_features
from hyperparameter_tuning import MAX_SEQ_LENGTH
import numpy as np

videos_df = load_features('./resources/JSON/')


#Split data into train & test dataframes while ensuring label representation
def split_data_with_label_representation(df):
    labels = df["label"].to_numpy()

  # For label-aware splitting
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)

  # Splitting based on label groups
    for train_index, test_index in sss.split(df, labels):
        train_X = df.iloc[train_index, :-1]  
        train_y = labels[train_index]
        test_X = df.iloc[test_index, :-1]
        test_y = labels[test_index]


    return train_X, test_X, train_y, test_y

def extend_keypoints(df):
    def extend_list(keypoints):
        current_length = len(keypoints)
        
        if current_length < MAX_SEQ_LENGTH:
            repeat_count = (MAX_SEQ_LENGTH - current_length) // current_length
            remainder = (MAX_SEQ_LENGTH - current_length) % current_length
            
            # Repeat the keypoints to match the repeat_count
            extended_keypoints = np.tile(keypoints, (repeat_count + 1, 1))
            # Add the remaining frames
            extended_keypoints = np.vstack((extended_keypoints, keypoints[:remainder]))
            
            # Truncate to MAX_SEQ_LENGTH if it's overflown
            return extended_keypoints[:MAX_SEQ_LENGTH]
        elif current_length > MAX_SEQ_LENGTH:
            # Truncate the keypoints to MAX_SEQ_LENGTH
            return keypoints[:MAX_SEQ_LENGTH]
        
        return keypoints

    df['keypoints'] = df['keypoints'].apply(lambda x: extend_list(np.array(x)))
    return df


def build_tensors(df):
    
  train_X, test_X, train_y, test_y = split_data_with_label_representation(df)

  train_data_padded = extend_keypoints(train_X)
  test_data_padded = extend_keypoints(test_X)

  train_X_tensor = []
  for video in train_data_padded['keypoints']:
    # Iterate through frames in the video
    frame_tensors = []
    for frame in video:
      # Convert each frame into a tensor
      frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
      frame_tensors.append(frame_tensor)
    # Append the list of frame tensors (representing the video) to the final list
    train_X_tensor.append(frame_tensors)

  test_X_tensor = []
  for video in test_data_padded['keypoints']:
    frame_tensors = []
    for frame in video:
      frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
      frame_tensors.append(frame_tensor)
    test_X_tensor.append(frame_tensors)

  return train_X_tensor, train_y, test_X_tensor, test_y
