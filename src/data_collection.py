import pandas as pd
import json
import os
import numpy as np
from typing import List, Dict, Any

def json_to_list(filepath: str) -> List[np.ndarray]:
    """
    Convert JSON file to a list of keypoints.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        List[np.ndarray]: List of keypoints arrays.
    """
    with open(filepath, 'r') as f:
        data: List[Dict[str, Any]] = json.load(f)

    keypoints_list: List[np.ndarray] = []
    # Creates a list of frames in a video
    for item in data:
        keypoints: List[float] = item["keypoints"]
        filtered_keypoints: List[float] = [point for i, point in enumerate(keypoints) if i % 3 != 2]  # Skip confidence score (Every 3rd value in list)
        keypoints_list.append(np.array(filtered_keypoints))
    
    return keypoints_list

def process_json_files(filepaths: List[str]) -> pd.DataFrame:
    """
    Process multiple JSON files into a DataFrame.

    Args:
        filepaths (List[str]): List of file paths to JSON files.

    Returns:
        pd.DataFrame: DataFrame containing keypoints.
    """
    df: pd.DataFrame = pd.DataFrame(columns=['keypoints'])
    for filepath in filepaths:
        keypoints_list: List[np.ndarray] = json_to_list(filepath)
        keypoints: np.ndarray = np.array(keypoints_list)
        # Data for a new row
        video_data: Dict[str, np.ndarray] = {'keypoints': keypoints}
        df = df._append(video_data, ignore_index=True)

    return df

def load_features(data_dir: str) -> pd.DataFrame:
    """
    Load features from JSON files in a directory.

    Args:
        data_dir (str): Directory containing JSON files.

    Returns:
        pd.DataFrame: DataFrame containing keypoints and labels.
    """
    filepaths: List[str] = []
    labels: List[str] = []
    # Creates lists of filepaths and their respective severity label
    for severity in ['NORMAL', 'MILD', 'MODERATE', 'SEVERE']:
        dir_path: str = os.path.join(data_dir, severity)
        for json_file in os.listdir(dir_path):
            file_path: str = os.path.join(dir_path, json_file)
            labels.append(severity)
            filepaths.append(file_path)
        
    df: pd.DataFrame = process_json_files(filepaths)

    # Adds the labels column to the final dataframe
    label_df: pd.DataFrame = pd.DataFrame(labels, columns=['label'])
    df = pd.concat([df, label_df], axis=1)
    
    return df

def create_csv() -> pd.DataFrame:
    """
    Create a CSV file from JSON files and save it.

    Returns:
        pd.DataFrame: DataFrame containing keypoints and labels.
    """
    df: pd.DataFrame = load_features('./resources/JSON/')
    
    # Convert arrays in keypoints column to JSON strings
    df['keypoints'] = df['keypoints'].apply(lambda x: json.dumps(x.tolist()))

    df.to_csv("./resources/labeled_keypoints.csv", index=False)
    return df

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing keypoints and labels.
    """
    df: pd.DataFrame = pd.read_csv(filepath)

    # Convert JSON strings back to lists of arrays
    df['keypoints'] = df['keypoints'].apply(lambda x: np.array(json.loads(x)))

    return df