import numpy as np
import cv2
import os

# Check if folder exists
if not os.path.exists('images'):
    os.makedirs('images')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
count = 0

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n enter user id ')
print("\n [INFO] Initializing face")


while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        
        cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('video',img)
    
    k = cv2.waitKey(20) & 0xff
    if k < 30:
        break
    
    elif count >= 30:
         break

print("\n next phase")
cam.release()
cv2.destroyAllWindows()
