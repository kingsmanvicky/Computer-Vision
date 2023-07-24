import cv2
import cv2 
import matplotlib.pyplot as plt



#Detecting faces by mounting the trained model
face_detector = cv2.CascadeClassifier("Face detection(webcam)/haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("lbph_classifier.yml")
width, height = 220,220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
video_capture = cv2.VideoCapture(0)


while True:
    #Capturing frame-by-frame
    ret, frame = video_capture.read()
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    face_detections = face_detector.detectMultiScale(image_gray)


    #Creating the Face box for visualization
    for (x, y, w, h) in face_detections:
        image_face = cv2.resize(image_gray[y:y+w, x:x+h],(width, height))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        Id, confidence = face_recognizer.predict(image_face)
        name = ""

        if Id == 1:
            name = "MG Vignesh"
        elif Id == 2:
            name = "Unknown"
        cv2.putText(frame, name, (x,y+(w+30)), font, 2, (0,0,255))
        cv2.putText(frame, str(confidence), (x,y+(h+50), font, 1, (0,0,255)))


    cv2.imshow( 'Display Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()