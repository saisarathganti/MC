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

# def send_paste(text_data, file_name):
# 	try:
# 		import requests
# 		key = "myyiWE5pZnwBzZX_gMTsQ9TLv2jcvaF-"
# 		url = "https://pastebin.com/api/api_post.php"
# 		args = {"api_dev_key": key, "api_paste_code":"test", "api_option":"paste"}
# 		login_data = {
# 		    'api_dev_key': key,
# 		    'api_user_name': 'sharugantiasu',
# 		    'api_user_password': '$N%vB8n3rAPnhN^'
# 		}

# 		data = {
# 		    'api_option': 'paste',
# 		    'api_dev_key':key,
# 		    'api_paste_code': text_data,
# 		    'api_paste_name': file_name,
# 		    'api_user_key': None,
# 		    }

# 		login = requests.post("https://pastebin.com/api/api_login.php", data=login_data)
# 		print("Login status: ", login.status_code if login.status_code != 200 else "OK/200")
# 		print("User token: ", login.text)
# 		data['api_user_key'] = login.text
		 
# 		r = requests.post("https://pastebin.com/api/api_post.php", data=data)
# 		print("Paste send: ", r.status_code if r.status_code != 200 else "OK/200")
# 		print("Paste URL: ", r.text)
# 	except Exception as ex:
# 		return

mappingTest = {}
mappingTrain = {}
local = False

training_feature_vector = {}
def create_penultimate_training_data():
	os.makedirs(TRAINING_FRAMES_PATH, exist_ok=True)
	for file_name in os.listdir(TRAINING_DATA_PATH):
		# if file_name=="frames":
		# 	continue
		file_path = os.path.join(TRAINING_DATA_PATH, file_name)
		if local:
			gesture = file_name.split("_")[0].split(".")[0]
		else:
			gesture = file_name.split("-")[1].split(".")[0]
		# print(gesture)
		frame_path = os.path.join(TRAINING_FRAMES_PATH, f"{gesture}.png")
		print(file_path, frame_path)
		frameExtractor(file_path, frame_path)
		frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
		mappingTrain[gesture] = HandShapeFeatureExtractor.get_instance().extract_feature(frame)


def create_penultimate_testing_data():
	os.makedirs(TESTING_FRAMES_PATH, exist_ok=True)
	for file_name in os.listdir(TESTING_DATA_PATH):
		file_path = os.path.join(TESTING_DATA_PATH, file_name)
		gesture = file_name.split("-")[2].split(".")[0]
		frame_path = os.path.join(TESTING_FRAMES_PATH, f"{gesture}.png")
		frameExtractor(file_path, frame_path)
		frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
		mappingTest[gesture] = HandShapeFeatureExtractor.get_instance().extract_feature(frame)

f = open("Results.csv", "w+")
writer = csv.writer(f)
data = [[i] for i in range(0,17)]
data += [[i] for i in range(0,17)]
data += [[i] for i in range(0,17)]
writer.writerows(data)
f.close()

# send_paste("line 80", "upd1")
# create_penultimate_training_data()
# send_paste("line 82", "upd2")
# create_penultimate_testing_data()
# send_paste("line 84", "upd3")
# gestureMappingRight = {"0":"0","1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9","DecreaseFanSpeed":"10","FanOn":"11","FanOff":"12", "IncreaseFanSpeed":"13", "LightOff": "14", "LightOn":"15", "SetThermo": "16"}
# # gestureMappingLeft={"0":"0","1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9","FanDown":"Decrease Fan Speed","FanOn":"FanOn","FanOff":"FanOff", "FanUp":"Increase Fan Speed", "LightOff": "LightOff", "LightOn":"LightOn", "SetThermo": "SetThermo"}

# mappingResult=[]
# for test_gesture_name, test_gesture_frame in mappingTest.items():
# 	result=float("inf")
# 	predicted_gesture = None
# 	for train_gesture_name, train_gesture_frame in mappingTrain.items():
# 		temp = spatial.distance.cosine(test_gesture_frame, train_gesture_frame)
# 		# print(i,a, temp, False)
# 		if temp < result:
# 			result=temp
# 			predicted_gesture = train_gesture_name
# 	mappingResult.append([gestureMappingRight[predicted_gesture]])


# # print(mappingResult)
# # mappingResult = [[1],[2],[3]]
# # f = open('Results.csv', 'w+')
# # writer = csv.writer(f)
# # writer.writerows(mappingResult)
# # f.close()




# # send_paste(os.listdir(os.getcwd()))
# # send_paste(str(list(os.walk("."))))

# send_paste(str(mappingResult), "mappingResult")