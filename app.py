from flask import Flask, render_template, request, redirect, Response
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import csv
import pandas as pd
import datapane as dp
import matplotlib.pyplot as plt
import os 


face_classifier = cv2.CascadeClassifier("G:\\PROJECTS\\Hack-O-HITS\\000FINAL\\report checking\\haarcascade_frontalface_default.xml")
classifier =load_model("G:\\PROJECTS\\Hack-O-HITS\\000FINAL\\report checking\\Emotion_Detection.h5")

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']



app = Flask(__name__)



def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    preds = classifier.predict(roi)[0]
                    label = class_labels[preds.argmax()]


                    input_variable = label
                    with open('C:\\Users\\VIMAL\\OneDrive\\hoh\\report.csv', 'a', newline = '') as csvfile:
                        my_writer = csv.writer(csvfile, delimiter = ' ')
                        my_writer.writerow(input_variable)
                    #sleep(2)


                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 3)  
                    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')



@ app.route('/emotion_detector')
def emotion_detector():
    title = 'Emotion Detector'
    return render_template('emo_det.html', title=title)


@ app.route('/report')
def report():
    title = 'Report'
    return render_template('report.html', title=title)



if __name__ == '__main__':
    app.run(debug=True)



























