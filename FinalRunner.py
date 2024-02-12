import cv2.data
from keras.models import model_from_json
import numpy as np
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
def extract_feature(image):
    features = np.array(image)
    features = features.reshape(1,48,48,1)
    return features/255.0
webcam=cv2.VideoCapture("Test1.mp4")
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im,1.3,5)
    try:
        for (x,y,w,h) in faces:
            image = gray[y:y+h,x:x+w]
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
            image = cv2.resize(image,(48,48))
            img = extract_feature(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im,'% s' %(prediction_label),(x-10,y-10),cv2.FONT_ITALIC,2,(255,0,0))
            cv2.imshow("GDSC",im)
            cv2.waitKey(27)
    except cv2.error:
        pass

