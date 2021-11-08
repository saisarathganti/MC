
import cv2
import numpy as np
import os
from handshape_feature_extractor import HandShapeFeatureExtractor
from scipy import spatial
model = HandShapeFeatureExtractor.get_instance()

def generatePenultimateTrainingLayer():
    inputPathName = "traindata"
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

def generatePenultimateTestingLayer():
    inputPathName = "test"
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


training_layer = generatePenultimateTrainingLayer()

testing_layer = generatePenultimateTestingLayer()

featureLabel = []
cosineSimilarity = []
for test in testing_layer:
    for train in training_layer:
        dist = spatial.distance.cosine(test, train)
        cosineSimilarity.append(dist)
    featureLabel.append(int(cosineSimilarity.index(min(cosineSimilarity))))
    cosineSimilarity = []

np.savetxt("Results.csv", featureLabel, fmt="%d")


