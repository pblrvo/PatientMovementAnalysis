import pandas as pd
import json
import os

def flatten_list(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def json_to_list(filepath):

  with open(filepath, 'r') as f:
    data = json.load(f)

  keypoints_list = []
  #Creates a keypoints list from all the frames in a video
  for item in data:
    keypoints = item["keypoints"]
    filtered_keypoints = [point for i, point in enumerate(keypoints) if i % 3 != 2]  # Skip confidence score (Every 3rd value in list)
    keypoints_list.append(filtered_keypoints)
  
  return keypoints_list


def process_json_files(filepaths, limit=122128):
  

  all_keypoints = []
  for filepath in filepaths:
    #Turn dataframe of video's keypoints into a flattened list
    json_keypoints_list = json_to_list(filepath)
    flattened_list = flatten_list(json_keypoints_list)

    # Add zero's to right end of list of shorter videos
    flattened_list = flattened_list[:limit] + [0] * (limit - len(flattened_list))

    all_keypoints.append(flattened_list)

  # Create a DataFrame from keypoints
  df = pd.DataFrame(all_keypoints)

  return df

def load_features(data_dir):
    filepaths = []
    labels = []
    #Creates lists of filepaths and their respective severity label
    for severity in ['NORMAL', 'MILD', 'MODERATE', 'SEVERE']:
        dir_path = os.path.join(data_dir, severity)
        for json_file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, json_file)
            labels.append(severity)
            filepaths.append(file_path)
        
    df = process_json_files(filepaths)

    #Adds the labels column to the final dataframe
    label_df = pd.DataFrame(labels, columns=['label'])
    df = pd.concat([df, label_df], axis=1)

    #Creates csv file from df
    df.to_csv("./resources/labeled_keypoints.csv", index=False)
    return 


load_features('./resources/JSON/')
