import cv2 
import matplotlib.pyplot as plt


#Image Processing
image = cv2.imread("Clocks detection\clock.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Detecting by mounting the trained model
clock_detector = cv2.CascadeClassifier("Clocks detection\Clocks Trained.xml")
clock_detections = clock_detector.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=1)


for (x, y, w, h) in clock_detections:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)


cv2.imshow( 'Display Image',image)
cv2.waitKey()