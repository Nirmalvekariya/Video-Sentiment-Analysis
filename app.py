import sys
import os
import numpy as np
from predictemt import pred, removeout, vidframe, ssimscore1
from flask import Flask, request, render_template, flash, redirect

from werkzeug.utils import secure_filename
import shutil
from tensorflow.keras.models import model_from_json
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
import io
import base64
import urllib


facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   #load face detection cascade file

app = Flask(__name__)
app.secret_key = 'some secret key'





@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' in request.files:

            f = request.files['file']  #getting uploaded video 
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)  #saving uploaded video

            result, face = vidframe(file_path) #running vidframe with the uploaded video
            os.remove(file_path)  #removing the video as we dont need it anymore
        else:
            result, face = vidframe(0)
        try:
            smileindex=result.count('happy')/len(result)  #smileIndex
            smileindex=round(smileindex,2)

        except:
            smileindex=0

        ssimscore=[ssimscore1(i,j) for i, j in zip(face[: -1],face[1 :])]  # calculating similarityscore for images
        if np.mean(ssimscore)<0.6:
        	posture="Not Good"
        else:
        	posture="Good"
        fig = plt.figure()     #matplotlib plot
        ax = fig.add_axes([0,0,1,1])
        ax.axis('equal')
        emotion = ['angry','disgust','fear', 'happy', 'sad']
        counts = [result.count('angry'),result.count('disgust'),result.count('fear'),result.count('happy'),result.count('sad')]
        ax.pie(counts, labels = emotion,autopct='%1.2f%%')   #adding pie chart
        img = io.BytesIO()
        plt.savefig(img, format='png')   #saving piechart
        img.seek(0)
        plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode()) #piechart object that can be returned to the html
        return render_template("predict.html", posture = posture, smileindex=smileindex, plot_url=plot_data) #returning all the three variable that can be displayed in html
    return None


if __name__ == '__main__':
    app.run(debug=False)
    app.secret_key = 'some secret key'
