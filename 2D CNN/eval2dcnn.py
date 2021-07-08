import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.xception import Xception
import h5py
import json
import cv2
import math
import logging
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

logging.basicConfig(level = logging.INFO)

sampling_rate = 5
sampled_frames = frame_stamps = []
top1_labels = top1_scores = []

def sampling_time_stamps(_sample_path):


	cap = cv2.VideoCapture(_sample_path)
	total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	logging.info('Total no. of frames in video:', total_frame_count)


	for i in range(sampling_rate):
	    val = round(total_frame_count/sampling_rate)*(i+1)
	    frame_stamps.append(val)

def sampling_frames():

	frameId , frame_count = 5, 0
	success,frame = cap.read()

	while success:

	    frame_count+=1
	    if frame_count in frame_stamps and frameId >= 1:
	        frame = cv2.resize(frame, (299,299))    
	        sampled_frames.append(frame)        
	        success,frame = cap.read()
	        frameId-=1
	    else:
	        success,frame = cap.read()
	        pass 


def generate_and_average_predictions():

	base_model = keras.applications.Xception(
	    weights='imagenet')  # Load weights pre-trained on ImageNet.

	for i in range(len(sampled_frames)):
	    img = sampled_frames[i]
	    x = image.img_to_array(img)
	    x = np.expand_dims(x, axis=0)
	    x = preprocess_input(x)
	    preds = base_model.predict(x)
	    print('Prediction level:', (i+1), decode_predictions(preds, top=5)[0])

	    top1_labels.append(decode_predictions(preds, top=1)[0][0][1])
	    top1_scores.append(decode_predictions(preds, top=1)[0][0][2])

	return top1_labels, top1_scores

def run():

	sampling_time_stamps(_sample_path)
	sampling_frames()
	labels, scores = generate_and_average_predictions()

	return labels, scores 

	