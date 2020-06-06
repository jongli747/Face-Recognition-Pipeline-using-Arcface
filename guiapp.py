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
flag = False #----> TRUE if you want to update face database

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,255,0),
                    1,
                    cv2.LINE_AA)
    return frame



def prepare_facebank(conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = ['Unknown']
    for path in conf.facebank_path.iterdir():
        print(path)
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    if img.size != (112, 112):
                        img = mtcnn.align(img)
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path/'facebank.pth')
    np.save(conf.facebank_path/'names', names)
    return embeddings, names

def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path/'facebank.pth')
    names = np.load(conf.facebank_path/'names.npy')
    return embeddings, names

mtcnn = MTCNN()

#-----------------> gui block with new code -------------------------!


det = faceDetector('./workspace/model/haarcascade_frontalface_default.xml')
file_path = './data/faces/'
dir_face = os.path.dirname(file_path)

#------------------- MTCNN MTCNN MTCNN ---------------------------!
# def isDir(dir_face, user_id):
#     for root, dirs, files in os.walk(dir_face):
#         for dir in dirs:
#             if dir == user_id:
#                 print('UREKA!')
#                 break
#             else:
#                 print('no dir!')
#
#
# def reg(path, user_id):
#     cap = cv2.VideoCapture(0)
#     while True:
#         isSuccess, frame = cap.read()
#
#         num_sample = 0
#         img = Image.fromarray(frame)
#         faces = mtcnn.align(img)
#         face = np.asarray(faces)
#         num_sample+=1
#
#         cv2.imwrite(f'{path}/{user_id}.{num_sample}.jpg', face)
#
#         # # bboxes = bboxes[:,:-1]
#         # # bboxes = bboxes.astype(int)
#         # # bboxes = bboxes = [-1,-1,1,1]
#         # # for bbox in bboxes:
#         # #     frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
#         # # cv2.imshow('arcface', frame)
#         if num_sample > 10:
#             break
#         cv2.imshow('faces', frame)
#         cv2.waitKey(100)
#------------------- MTCNN MTCNN MTCNN ---------------------------!

def registration(path, user_id):
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
        cv2.imshow('faces',frame)
        cv2.waitKey(1)


        if num_sample >10:
            break


def gui():
    opt = pyautogui.confirm(text = 'Choose an option', title = 'Face Recognition System', buttons = ['Live Demo', 'Registration', 'Exit'])

    if opt == 'Live Demo':
        print('Go live!')
        import faces_test
    if opt == 'Registration':
        opt = pyautogui.confirm(text = "Please look at the Camera.\nTrun your head a little while capturing.\nPlease add one face at a time.\nClick 'Ready' when you are.",
        title = 'Register', buttons = ['Ready', 'Cancel'])

        if opt == 'Ready':
            #print('Re train with new picture!')
            user_id = pyautogui.prompt(text = "Enter User ID.\n\nnote: First letter upper case.", title = 'Registration', default = 'none')
            user_id = user_id.capitalize()
            print(user_id)
            d = os.path.join(dir_face,user_id)
            if not os.path.exists(d):
                os.mkdir(d)
                print("Directory created!")
            registration(d,user_id)
            print(d)
        if opt == 'Cancel':
            print('cancel!')

    if opt == 'Exit':
        print('Quit window')

gui()


