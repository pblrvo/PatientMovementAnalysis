from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from data_collection import load_csv
from hyperparameter_tuning import MAX_SEQ_LENGTH
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Split data into keypoints and labels
def split_data_and_labels(df):
    labels = df["label"].to_numpy()
    keypoints_df = df.drop(columns=["label"])

    return keypoints_df, labels

# Extend keypoints to MAX_SEQ_LENGTH
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

# Standardize keypoints
def standardize_keypoints(df):
    scaler = StandardScaler()
    all_standardized_keypoints = []
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        keypoints = df.loc[i, 'keypoints']

        standardized_video_keypoints = []
        for frame_keypoints in keypoints:
            # Reshape to 2D for StandardScaler
            frame_keypoints = frame_keypoints.reshape(-1, 1)  # Reshape to (n_samples, n_features)
            # Standardize using StandardScaler
            standardized_frame_keypoints = scaler.fit_transform(frame_keypoints)
            # Reshape back to original frame format
            standardized_frame_keypoints = standardized_frame_keypoints.reshape(-1)
            standardized_video_keypoints.append(standardized_frame_keypoints)

        all_standardized_keypoints.append(standardized_video_keypoints)

    return all_standardized_keypoints  # Return the list of standardized video keypoints

# Build tensors from dataframe
def build_tensors(df):
    df_data, df_labels = split_data_and_labels(df)

    le = LabelEncoder()
    df_labels = le.fit_transform(df_labels)

    df_data = extend_keypoints(df_data)
    data_extended = standardize_keypoints(df_data)

    data_tensors = []
    for video in data_extended:
        # Iterate through frames in the video
        frame_tensors = []
        for frame in video:
            # Convert each frame into a tensor
            frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
            frame_tensors.append(frame_tensor)
        # Append the list of frame tensors (representing the video) to the final list
        data_tensors.append(frame_tensors)

    return np.array(data_tensors), df_labels, le.classes_