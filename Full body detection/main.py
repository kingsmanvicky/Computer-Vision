import cv2 
import matplotlib.pyplot as plt


#Image Processing
image = cv2.imread("Full body detection\people3.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Detecting body by mounting the trained model
body_detector = cv2.CascadeClassifier("Full body detection\Full body trained.xml")
body_detections = body_detector.detectMultiScale(image_gray, scaleFactor= 1.05, minNeighbors=5, minSize=(50,50))


#Creating the body box for visualization
for (x, y, w, h) in body_detections:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)


cv2.imshow( 'Display Image',image)
cv2.waitKey()