import cv2 
import matplotlib.pyplot as plt



#Detecting faces by mounting the trained model
face_detector = cv2.CascadeClassifier("Face detection(webcam)/haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)


while True:
    #Capturing frame-by-frame
    ret, frame = video_capture.read()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    face_detections = face_detector.detectMultiScale(image_gray)


    #Creating the Face box for visualization
    for (x, y, w, h) in face_detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)


    cv2.imshow( 'Display Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()