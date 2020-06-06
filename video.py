import cv2
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


dont_know = 0

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
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


#print(summary(learner.model, input_size=(3,112,112)))


print(targets.shape, names.shape)

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,500)

cap = cv2.VideoCapture('./sample_video/baker_bhai_cut.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC, 0)
fps = cap.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter('./output_video/baker_bhai_cut_output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

i = 0



while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:
#             image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            image = Image.fromarray(frame)
            try:
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            except:
                bboxes = []
                faces = []
            if len(bboxes) == 0:
                print('no face')
                continue
            else:
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice
                results, score = learner.infer(conf, faces, targets, True)
                for idx,bbox in enumerate(bboxes):
                    # if args.score:
                    #     frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    # else:
                    frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            video_writer.write(frame)
        else:
            break
        # if  != 0:
        #     i += 1
        #     if i % 25 == 0:
        #         print('{} second'.format(i // 25))
        #     if i > 25 * args.duration:
        #         break
cap.release()
video_writer.release()
