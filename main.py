from src.data_collection import load_csv
from utils.download_data import download_file_from_drive, decompress_file

if __name__ == '__main__':
    download_file_from_drive('1rqwyCpqf82_zhyD9Jay4m7Y8HXJmW4pJ', './resources/labeled_keypoints.csv.zip')
    password = ''
    decompress_file('./resources/labeled_keypoints.csv.zip', './resources', password)
    load_csv('./resources/labeled_keypoints.csv')
    #setup.preprocess()
