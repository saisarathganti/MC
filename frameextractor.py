# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
#code to get the key frame from the video and save it as a png file.

import cv2
import os
import numpy as np
#videopath : path of the video file
#frames_path: path of the directory to which the frames are saved
#count: to assign the video order to the frane.
# 1700 is good with 11% accuracy
# 1920, 1080 original
# h=1700
# w=900
def frameExtractor(videopath, output_frame_path):
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no= int(video_length/2)
    #print("Extracting frame..\n")
    cap.set(1,frame_no)
    ret,frame=cap.read()
    # print(frame.shape)
    # center0 = frame.shape[0] / 2
    # center1 = frame.shape[1] / 2
    # x = center1 - w/2
    # y = center0 - h/2

    # crop_img = frame[int(y):int(y+h), int(x):int(x+w)]
    # # print(crop_img.shape)
    # crop_img = cv2.resize(crop_img, (200, 200))

    cv2.imwrite(output_frame_path, frame)