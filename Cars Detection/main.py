import cv2 
import matplotlib.pyplot as plt


#Image Processing
image = cv2.imread("Cars Detection\car.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Detecting by mounting the trained model
car_detector = cv2.CascadeClassifier("Cars Detection\Cars trained.xml")
car_detections = car_detector.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=5)



for (x, y, w, h) in car_detections:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)


cv2.imshow( 'Display Image',image)
cv2.waitKey()