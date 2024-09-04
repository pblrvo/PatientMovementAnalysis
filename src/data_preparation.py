import tensorflow as tf
from src.hyperparameter_tuning import MAX_SEQ_LENGTH
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pickle import dump, load
from typing import Tuple, List, Any

def split_data_and_labels(videos_dataframe: Any) -> Tuple[Any, np.ndarray]:
    """
    Split the dataframe into data and labels.

    Args:
        videos_dataframe (Any): DataFrame containing video data and labels.

    Returns:
        Tuple[Any, np.ndarray]: DataFrame without labels and array of labels.
    """
    labels: np.ndarray = videos_dataframe["label"].to_numpy()
    dataframe_data: Any = videos_dataframe.drop(columns=["label"])

    return dataframe_data, labels

def extend_keypoints(df: Any) -> Any:
    """
    Extend keypoints to match MAX_SEQ_LENGTH.

    Args:
        df (Any): DataFrame containing keypoints.

    Returns:
        Any: DataFrame with extended keypoints.
    """
    def extend_list(keypoints: np.ndarray) -> np.ndarray:
        current_length: int = len(keypoints)

        if current_length < MAX_SEQ_LENGTH:
            repeat_count: int = (MAX_SEQ_LENGTH - current_length) // current_length
            remainder: int = (MAX_SEQ_LENGTH - current_length) % current_length

            extended_keypoints: np.ndarray = np.tile(keypoints, (repeat_count + 1, 1))
            extended_keypoints = np.vstack((extended_keypoints, keypoints[:remainder]))

            return extended_keypoints[:MAX_SEQ_LENGTH]
        elif current_length > MAX_SEQ_LENGTH:
            return keypoints[:MAX_SEQ_LENGTH]

        return keypoints

    df['keypoints'] = df['keypoints'].apply(lambda x: extend_list(np.array(x)))
    return df

def standardize_keypoints(df: Any, is_scaler_saved: bool = False) -> List[np.ndarray]:
    """
    Standardize keypoints using StandardScaler.

    Args:
        df (Any): DataFrame containing keypoints.
        is_scaler_saved (bool, optional): Whether to load a saved scaler. Defaults to False.

    Returns:
        List[np.ndarray]: List of standardized keypoints.
    """
    if not is_scaler_saved:
        scaler: StandardScaler = StandardScaler()
    else:
        scaler: StandardScaler = load(open('results/scaler.pkl', 'rb'))

    all_standardized_keypoints: List[np.ndarray] = []
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        keypoints: np.ndarray = df.loc[i, 'keypoints']
        standardized_video_keypoints: List[np.ndarray] = []
        for frame_keypoints in keypoints:
            frame_keypoints = frame_keypoints.reshape(-1, 1)
            standardized_frame_keypoints: np.ndarray = scaler.fit_transform(frame_keypoints)
            standardized_frame_keypoints = standardized_frame_keypoints.reshape(-1)
            standardized_video_keypoints.append(standardized_frame_keypoints)
        all_standardized_keypoints.append(standardized_video_keypoints)

    if not is_scaler_saved:
        dump(scaler, open('results/scaler.pkl', 'wb'))

    return all_standardized_keypoints

def build_tensors(videos_dataframe: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build tensors from the dataframe.

    Args:
        videos_dataframe (Any): DataFrame containing video data and labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Train data, train labels, test data, test labels, and label classes.
    """
    dataframe_data, labels = split_data_and_labels(videos_dataframe)

    train_data, test_data, train_labels, test_labels = train_test_split(dataframe_data, labels, test_size=0.15, random_state=42)

    le: LabelEncoder = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.fit_transform(test_labels)

    train_data = extend_keypoints(train_data)
    train_data = standardize_keypoints(train_data)
    test_data = extend_keypoints(test_data)
    test_data = standardize_keypoints(test_data)

    train_tensors: List[List[tf.Tensor]] = []
    for video in train_data:
        frame_tensors: List[tf.Tensor] = []
        for frame in video:
            frame_tensor: tf.Tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
            frame_tensors.append(frame_tensor)
        train_tensors.append(frame_tensors)

    test_tensors: List[List[tf.Tensor]] = []
    for video in test_data:
        frame_tensors: List[tf.Tensor] = []
        for frame in video:
            frame_tensor: tf.Tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
            frame_tensors.append(frame_tensor)
        test_tensors.append(frame_tensors)

    return np.array(train_tensors), train_labels, np.array(test_tensors), test_labels, le.classes_

def build_tensors_for_test(videos_dataframe: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build tensors from the test dataframe.

    Args:
        videos_dataframe (Any): DataFrame containing video data and labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Test data and test labels.
    """
    dataframe_data, labels = split_data_and_labels(videos_dataframe)

    le: LabelEncoder = LabelEncoder()
    test_labels = le.fit_transform(labels)

    test_data = extend_keypoints(dataframe_data)
    test_data = standardize_keypoints(test_data, is_scaler_saved=True)

    test_tensors: List[List[tf.Tensor]] = []
    for video in test_data:
        frame_tensors: List[tf.Tensor] = []
        for frame in video:
            frame_tensor: tf.Tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
            frame_tensors.append(frame_tensor)
        test_tensors.append(frame_tensors)

    return np.array(test_tensors), test_labels