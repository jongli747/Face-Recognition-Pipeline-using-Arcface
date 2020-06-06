import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
import pyautogui
from config import get_config
from torchvision import transforms as trans
from model import l2_norm
import numpy as np
from FaceDetector import *
from mtcnn import MTCNN
from Learner import face_learner
#from utils import load_facebank, draw_box_name, prepare_facebank
from torchsummary import summary
from datetime import datetime
import os

det = faceDetector('./workspace/model/haarcascade_frontalface_default.xml')
file_path = './data/faces/'
dir_face = os.path.dirname(file_path)


class reg:

	def registration(self,path, user_id):

	    num_sample = 0
	    cap = cv2.VideoCapture(0)
	    while True:
	        _,frame = cap.read()
	        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	        fd = det.detect(gray)
	        for(x,y,w,h) in fd:
	            roi = frame[y:y+h, x:x+w]
	            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	            roi = cv2.resize(roi, (112,112))
	            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	            num_sample = num_sample + 1
	            cv2.imwrite(f'{path}/{user_id}_{num_sample}.png', roi)
	            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,225), 2)
	            cv2.waitKey(100)
	        # cv2.imshow('faces',frame)
	        cv2.waitKey(1)


	        if num_sample >10:
	            break

	def main(self,s):
		user_id = s.capitalize()
		print(user_id)
		d = os.path.join(dir_face,user_id)
		if not os.path.exists(d):
			os.mkdir(d)
			print('dir done!')
		return d,user_id
		# registration(d,user_id)

		
		


