#!/usr/bin/env python
# coding: utf-8

# In[61]:


from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import math
import tensorflow as tf
from tensorflow.keras.preprocessing import image
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
from matplotlib import pyplot as plt
import os
import shutil
from skimage.measure import compare_ssim


# In[62]:



# In[63]:


with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_weights.h5")
loaded_model._make_predict_function()
label_to_text = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4: 'sad'}


# In[64]:


def pred(img_path):  
    label_to_text = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4: 'sad'}  
    img=cv2.imread(img_path)
    gray_fr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_rects = facec.detectMultiScale(gray_fr, scaleFactor = 1.2, minNeighbors = 5)
    if len(faces_rects)!=0:
        for (x, y, w, h) in faces_rects:
            fc = gray_fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        img = image.img_to_array(roi)
        img = img/255
        img = np.expand_dims(img, axis=0)
        return label_to_text[np.argmax(loaded_model.predict(img))],img
    else:
        return 0,0


# In[65]:


def removeout():
    shutil.rmtree('output/')


# In[66]:


def vidframe(vidname):
	if os.path.exists('output'):
		removeout()
	os.mkdir('output')
	cap = cv2.VideoCapture(vidname)
	frameRate=cap.get(5)
	count = 0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename ="output/frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()
    result=[]
    face=[]
    for filename in os.listdir("output"):
        a,b = pred("output/"+filename)
        result.append(a)
        face.append(b)
    result=[x for x in result if x!=0]
    face=[x for x in face if len(str(x))>1]
    print ("Done!")
    return result, face


# In[68]:


def rmse(im1,im2):
    err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    err /= float(im1.shape[0] * im2.shape[1])
    return err


# In[106]:


def ssimscore1(im1,im2):
    im1=im1.reshape(48, 48, 1).astype('float32')
    im2=im2.reshape(48, 48, 1).astype('float32')
    (score, diff) = compare_ssim(im1, im2, full=True,multichannel=True)
    return score
    





