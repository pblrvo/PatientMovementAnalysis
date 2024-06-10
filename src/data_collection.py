import pandas as pd
import json
import os
import numpy as np


def json_to_list(filepath):

  with open(filepath, 'r') as f:
    data = json.load(f)

  keypoints_list = []
  #Creates a keypoints list from all the frames in a video
  for item in data:
    keypoints = item["keypoints"]
    filtered_keypoints = [point for i, point in enumerate(keypoints) if i % 3 != 2]  # Skip confidence score (Every 3rd value in list)
    keypoints_list.append(np.array(filtered_keypoints))
  
  return keypoints_list


def process_json_files(filepaths):
  df = pd.DataFrame(columns=['keypoints'])
  for filepath in filepaths:
    #Turn dataframe of video's keypoints into a flattened list
    keypoints_list = json_to_list(filepath)
    keypoints = np.array(keypoints_list)
    # Data for a new row
    video_data = {'keypoints': keypoints}
    df = df._append(video_data, ignore_index=True)

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
    
    return df


def create_csv():
   df = load_features('./resources/JSON/')
   df.to_csv("./resources/labeled_keypoints.csv", index=False)
   return