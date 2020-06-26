# Video-Sentiment-Analysis

----->  Run the app using python app.py
----->  Upload or capture a video for sentiment analysis.
----->  Get a result using the piechart.


•	Training the model for Facial Expression Detection
1)	File name EmotionDetection.ipynb
2)	DataSet used is Fer2013 data set which contains more than 35K images and 7 types of emotions. For this project we will be considering 5 emotions.
3)	In this data set 48*48 pixel grayscale images are flattened. So, we will reshape the data into 48*48 pixels. #LINE 30
4)	I have split the data for training and validation using sklerarn’s traintestsplit method and then have normalized the dataset. #Line 33/34
5)	Using ImageDataGenerator it will augment the datasets with its flip images aswell.
6)	Then for training the model I have used four convolution and two dense layers.
7)	In model compilation Ihave used Adam optimizer.
8)	I have also used callbacks that will save the model when high validation accuracy is achieved
9)	I have the model in json format.

•	Video analysis
1.	Here the main file is predictemt.py
2.	There are four function in this file that will be used in our project
3.	Pred():
a.	It takes image as an input and then it will convert it into grayscale
b.	Then using OpenCV’s cascade classifier it will detect the face of the person.
c.	Then we will only use the face for predicting and if the face is not found in the image than It will return 0.
d.	Then face image will be resized accourding to the size acceptable by our model.
4.	Vidframe():
a.	It takes an video as a input and then it will generate frames for that video.
b.	Each frame will be named and stored in output folder.
c.	Then after for every image it will run the pred function and will return the emotion and face of an person if found. 

5.	Ssimscore1() and removeout():
a.	Ssimscore1 is used to compare the faces that are returned by the pred function
b.	Removeout is used for removing the output directory.

•	App.py flask app
1.	In this file first the CNN Emotion detection model is loaded that is stored in json file and  also assigned the weights
2.	Here the main function is upload that is used for uploading video
3.	Then the vidframe function is run on the video that will return the emotion and faces
4.	SmileIndex is calculated by dividing total happy images to total images
5.	Ssimscore is calculated for every faces and if the score is less than 0.6 than we will say that the posture is not good.
6.	Then after we will create a pie chart for the emotions and we will return that pie chart to our predict template.
