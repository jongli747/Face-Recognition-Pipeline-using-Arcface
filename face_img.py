import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from torchvision import transforms as trans
from model import l2_norm
import numpy as np
from mtcnn import MTCNN
from Learner import face_learner
#from utils import load_facebank, draw_box_name, prepare_facebank
from torchsummary import summary
from datetime import datetime

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



conf = get_config(False)

learner = face_learner(conf,True)
#print(conf)

mtcnn = MTCNN()
print("loaded")

learner.load_state(conf, 'ir_se50.pth', False, True)

if flag:
    targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = True)
else:
    targets, names = load_facebank(conf)

print(targets.shape, names.shape)
#
#
# #print(summary(learner.model, input_size=(3,112,112)))
#
#
# #print(emb.shape, target.shape)
#
#
################# INFERENCE #################################################
f = '/home/nabeel/face/arcface_pipeline/bakerbhai.jpg'
img = Image.open(f)
# img = img.convert('RGB')
# img = np.asarray(img)
image = np.asarray(img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


bboxes, faces = mtcnn.align_multi(img, conf.face_limit, conf.min_face_size)
bboxes = bboxes[:,:-1]
bboxes = bboxes.astype(int)
#print(bboxes)
bboxes = bboxes + [-1,-1,1,1]
s = time.time()
results, score = learner.infer(conf, faces, targets, tta = True)
e = time.time() - s
print(e)
for idx,bbox in enumerate(bboxes):
    image = draw_box_name(bbox, names[results[idx] + 1], image)
ret,jpeg = cv2.imencode('.jpg',image)
cv2.imshow('arc face', image)
cv2.waitKey(5000)
# print(results)
# print(bboxes)




# cap = cv2.VideoCapture(0)
# cap.set(3,500)
# cap.set(4,500)
#
#
#
# while cap.isOpened():
#     isSuccess,frame = cap.read()
#
#     if isSuccess:
#
#         try:
#
# #           image = Image.fromarray(frame[...,::-1]) #bgr to rgb
#             print("Hello")
#             image = Image.fromarray(frame)
#
#             #print(type(image))
#             bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
#             print(type(bboxes))
#             bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
#             bboxes = bboxes.astype(int)
#             bboxes = bboxes + [-1,-1,1,1] # personal choice
#             results, score = learner.infer(conf, faces, targets, tta = True)
#             print(score[0])
#             for idx,bbox in enumerate(bboxes):
#                 frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
#         except:
#             print('Unknown Face')
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         #return jpeg.tostring()
#         cv2.imshow('Arc Face Recognizer', frame)
#
#
#     if cv2.waitKey(1)&0xFF == ord('q'):
#         break
#
# cap.release()
#
# cv2.destroyAllWindows()
