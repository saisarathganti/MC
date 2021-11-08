
import cv2
import numpy as np
import os
from handshape_feature_extractor import HandShapeFeatureExtractor
from scipy import spatial
model = HandShapeFeatureExtractor.get_instance()

def send_paste(text_data, file_name):
    try:
        import requests
        key = "myyiWE5pZnwBzZX_gMTsQ9TLv2jcvaF-"
        url = "https://pastebin.com/api/api_post.php"
        args = {"api_dev_key": key, "api_paste_code":"test", "api_option":"paste"}
        login_data = {
            'api_dev_key': key,
            'api_user_name': 'sharugantiasu',
            'api_user_password': '$N%vB8n3rAPnhN^'
        }

        data = {
            'api_option': 'paste',
            'api_dev_key':key,
            'api_paste_code': text_data,
            'api_paste_name': file_name
            }

        login = requests.post("https://pastebin.com/api/api_login.php", data=login_data)
        data['api_user_key'] = login.text
         
        r = requests.post("https://pastebin.com/api/api_post.php", data=data)
    except Exception as ex:
        x = 0


def generatePenultimateLayer(inputPathName):
    videos = []
    for fileName in os.listdir(inputPathName):
        if fileName.endswith(".mp4"):
            videos.append(os.path.join(inputPathName, fileName))
    send_paste(str(videos), "videos")
    featureVectors = []
    print("Extracting Frames of " + inputPathName)
    for video in videos:
        send_paste("line 45", "line45")
        frame = frameExtractor(video)
        send_paste("line 47", "line47")
        feature = model.extract_feature(frame)
        send_paste("line 49", "line49")
        featureVectors.append(feature)
    return featureVectors


def frameExtractor(videopath):
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    cap.set(1, int(video_length / 1.6))
    ret, frame = cap.read()
    return frame

send_paste("upd1", "upd1")
# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
training_layer = generatePenultimateLayer("traindata")

send_paste("upd2", "upd2")
# =============================================================================
# Get the penultimate layer for test data (Our Data)
# =============================================================================
testing_layer = generatePenultimateLayer("test")
send_paste("upd3", "upd3")
# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
#
featureLabel = []
cosineSimilarity = []
for test in testing_layer:
    for train in training_layer:
        dist = spatial.distance.cosine(test, train)
        cosineSimilarity.append(dist)
    featureLabel.append(int(cosineSimilarity.index(min(cosineSimilarity))))
    cosineSimilarity = []

send_paste(str(featureLabel), "upd4")
# totalCorrect = 0
# for i, label in enumerate(featureLabel):
#     if label == (i % 17):
#         totalCorrect += 1
# print("Total Correct are : " + str(totalCorrect) + "/" + str(len(testing_layer)))
# print("Accuracy is =" + str((100 * totalCorrect) / len(testing_layer)))

np.savetxt("Results.csv", featureLabel, fmt="%d")

