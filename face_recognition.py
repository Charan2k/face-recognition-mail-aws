import cv2
import numpy as np
from helpers import email, whatsapp, launch_aws, is_path


recognizer = cv2.face.LBPHFaceRecognizer_create()

is_path("saved_model/")
recognizer.read('saved_model/s_model.yml')

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:

        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        confidence = int(100 - confidence + 35)

        if Id == 1:
            Id = f'Charan {round(confidence)}%'
            if confidence > 80:
                email()
                print("Email Sent")
                whatsapp()
                exit(0)

        elif Id == 2 :
            Id = f"Sai {round(confidence)}%"
            if confidence > 80:
                launch_aws()
                exit(0)
        else:
            pass

        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)
    cv2.imshow('im',im) 
    
    if cv2.waitKey(10) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
