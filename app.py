import sys
import os
import numpy as np
from predictemt import pred, removeout, vidframe, ssimscore1
from flask import Flask, request, render_template

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


facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app = Flask(__name__)


with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_weights.h5")
loaded_model._make_predict_function()
label_to_text = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4: 'sad'}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        result, face = vidframe(file_path)

        smileindex=result.count('happy')/len(result)
        ssimscore=[ssimscore1(i,j) for i, j in zip(face[: -1],face[1 :])]
        if np.mean(ssimscore)<0.6:
        	posture="Not Good"
        else:
        	posture="Good"
        	fig = plt.figure()
        	ax = fig.add_axes([0,0,1,1])
        	ax.axis('equal')
        	emotion = ['angry','disgust','fear', 'happy', 'sad']
        	counts = [result.count('angry'),result.count('disgust'),result.count('fear'),result.count('happy'),result.count('sad')]
        	ax.pie(counts, labels = emotion,autopct='%1.2f%%')
        	img = io.BytesIO()
        	plt.savefig(img, format='png')
        	img.seek(0)
        	plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())
        	return render_template("predict.html", posture = posture, smileindex=smileindex, plot_url=plot_data) 
    return None


if __name__ == '__main__':
    app.run(debug=True)