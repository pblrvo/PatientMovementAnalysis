import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os


def load_keypoints_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    keypoints_list = []
    for element in data:
        keypoints = np.array(element['keypoints']).reshape(-1, 3)
        keypoints_list.append(keypoints)

    return keypoints_list

def filter_keypoints_by_confidence(keypoints_list, min_confidence=0.25):
    filtered_keypoints_list = []
    for keypoints in keypoints_list:
        filtered_keypoints = keypoints[keypoints[:, 2] > min_confidence]
        filtered_keypoints_list.append(filtered_keypoints)
    return filtered_keypoints_list

def extract_features(keypoints_list):
    total_nose_to_foot_distance = 0
    total_wrist_steadiness = 0
    total_nose_to_wrist_distance = 0
    total_arm_swing_symmetry = 0
    max_ankle_height = 0
    total_step_length = 0
    num_frames = 0

    #indices of keypoints
    nose_idx = 0
    left_wrist_idx = 9
    right_wrist_idx = 10
    left_ankle_idx = 15
    right_ankle_idx = 16

    for keypoints in keypoints_list:
        if keypoints.shape[0] < 17:
            continue
       
        nose = keypoints[nose_idx]
        left_wrist = keypoints[left_wrist_idx]
        right_wrist = keypoints[right_wrist_idx]
        left_ankle = keypoints[left_ankle_idx]
        right_ankle = keypoints[right_ankle_idx]

        # features 
        nose_to_foot_distance = np.linalg.norm(nose[:2] - ((left_ankle[:2] + right_ankle[:2]) / 2)) # Distance between the nose and the approximate center of the feet, indicating posture
        wrist_steadiness = np.mean([left_wrist[2], right_wrist[2]]) # Average confidence score of the wrists, indicating arm steadiness (tremors)
        nose_to_wrist_distance = np.mean([np.linalg.norm(nose[:2] - left_wrist[:2]), np.linalg.norm(nose[:2] - right_wrist[:2])]) # Average distance between the nose and the wrists, indicating arm swing
        arm_swing_symmetry = np.abs(np.linalg.norm(nose[:2] - left_wrist[:2]) / np.linalg.norm(nose[:2] - right_wrist[:2])) # Ratio of the distances between the nose and the left and right wrists, indicating arm swing symmetry
        max_ankle_height = np.max([left_ankle[1], right_ankle[1]])# Maximum height of the ankles, indicating step height
        step_length = np.linalg.norm(left_ankle[:2] - right_ankle[:2]) # Distance between the left and right ankles, indicating step length

        #Update Values
        total_nose_to_foot_distance += nose_to_foot_distance
        total_wrist_steadiness += wrist_steadiness
        total_nose_to_wrist_distance += nose_to_wrist_distance
        total_arm_swing_symmetry += arm_swing_symmetry
        max_ankle_height = max(max_ankle_height, np.max([left_ankle[1], right_ankle[1]]))
        total_step_length += step_length
        num_frames += 1

    if num_frames > 0:
        avg_nose_to_foot_distance = total_nose_to_foot_distance / num_frames
        avg_wrist_steadiness = total_wrist_steadiness / num_frames
        avg_nose_to_wrist_distance = total_nose_to_wrist_distance / num_frames
        avg_arm_swing_symmetry = total_arm_swing_symmetry / num_frames
        avg_step_length = total_step_length / num_frames
    else:
        return None
    
    return np.array([
        avg_nose_to_foot_distance,
        avg_wrist_steadiness,
        avg_nose_to_wrist_distance,
        avg_arm_swing_symmetry,
        max_ankle_height,
        avg_step_length
    ])

def load_data(data_dir):
    features = []
    labels = []

    for severity in ['NORMAL', 'MILD', 'MODERATE', 'SEVERE']:
        dir_path = os.path.join(data_dir, severity)

        for json_file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, json_file)
            keypoints = load_keypoints_from_json(file_path)
            filtered_keypoints = filter_keypoints_by_confidence(keypoints)

            if len(keypoints) > 0:
                feature = extract_features(keypoints)
                if feature is not None:
                    features.append(feature)
                labels.append(severity)

    return np.array(features), np.array(labels)

features, labels = load_data('./JSON/')


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
