import cv2
import numpy as np
import os
import sys
import tensorflow as tf
## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from scipy import spatial
import csv

workingDir = "" #sys.path[0]
TRAINING_DATA_PATH = workingDir + "traindata"
TESTING_DATA_PATH = workingDir + "test"

TRAINING_FRAMES_PATH = workingDir + "frames/train"
TESTING_FRAMES_PATH = workingDir + "frames/test"


mappingTest = {}
mappingTrain = {}
local = False

def get_middle_frame(videopath):
	capture = cv2.VideoCapture(videopath)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    capture.set(1, int(length / 1.6))
    ret, frame = capture.read()
    return frame

training_feature_vector = {}
def create_penultimate_training_data():
	for file_name in os.listdir(TRAINING_DATA_PATH):
		# send_paste(file_name, file_name)
		file_path = os.path.join(TRAINING_DATA_PATH, file_name)
		if local:
			gesture = file_name.split("_")[0].split(".")[0]
		else:
			gesture = file_name.split("-")[1].split(".")[0]

		frame = get_middle_frame(file_path)
		mappingTrain[gesture] = HandShapeFeatureExtractor.get_instance().extract_feature(frame)


def create_penultimate_testing_data():
	for file_name in os.listdir(TESTING_DATA_PATH):
		file_path = os.path.join(TESTING_DATA_PATH, file_name)
		gesture = file_name.split("-")[2].split(".")[0]
		frame_path = os.path.join(TESTING_FRAMES_PATH, gesture+".png")
		frame = get_middle_frame(file_path)
		mappingTest[gesture] = HandShapeFeatureExtractor.get_instance().extract_feature(frame)

# f = open("Results.csv", "w+")
# writer = csv.writer(f)
# data = [[i] for i in range(0,17)]
# data += [[i] for i in range(0,17)]
# data += [[i] for i in range(0,17)]
# writer.writerows(data)
# f.close()
# import platform
# send_paste(platform.python_version(), "PY_VERSION")
# send_paste("line 80", "upd1")
create_penultimate_training_data()
# send_paste("line 82", "upd2")
create_penultimate_testing_data()
# send_paste("line 84", "upd3")
gestureMappingRight = {"0":"0","1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9","DecreaseFanSpeed":"10","FanOn":"11","FanOff":"12", "IncreaseFanSpeed":"13", "LightOff": "14", "LightOn":"15", "SetThermo": "16"}
# gestureMappingLeft={"0":"0","1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9","FanDown":"Decrease Fan Speed","FanOn":"FanOn","FanOff":"FanOff", "FanUp":"Increase Fan Speed", "LightOff": "LightOff", "LightOn":"LightOn", "SetThermo": "SetThermo"}

mappingResult=[]
for test_gesture_name, test_gesture_frame in mappingTest.items():
	result=float("inf")
	predicted_gesture = None
	for train_gesture_name, train_gesture_frame in mappingTrain.items():
		temp = spatial.distance.cosine(test_gesture_frame, train_gesture_frame)
		# print(i,a, temp, False)
		if temp < result:
			result=temp
			predicted_gesture = train_gesture_name
	mappingResult.append([gestureMappingRight[predicted_gesture]])


print(mappingResult)
f = open('Results.csv', 'w+')
writer = csv.writer(f)
writer.writerows(mappingResult)
f.close()




# # send_paste(os.listdir(os.getcwd()))
# # send_paste(str(list(os.walk("."))))

send_paste(str(mappingResult), "mappingResult")