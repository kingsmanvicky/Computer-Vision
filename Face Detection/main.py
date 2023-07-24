import cv2 
import matplotlib.pyplot as plt


#Image Processing
image = cv2.imread("people1.jpg")
# image = cv2.resize(image, (800, 600))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Detecting faces by mounting the trained model
face_detector = cv2.CascadeClassifier("Trained Front Face.xml")
face_detections = face_detector.detectMultiScale(image_gray, scaleFactor= 1.3, minSize= (30,30))


#Detecting eyes by mounting the trained model
eye_detector = cv2.CascadeClassifier("Trained Eye.xml")
eye_detections = eye_detector.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors= 10, maxSize = (43,43))


#Creating the Face box for visualization
for (x, y, w, h) in face_detections:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
for (x, y, w, h) in eye_detections:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)


cv2.imshow( 'Display Image',image)
cv2.waitKey()