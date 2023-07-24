import dlib
import cv2


#Loading the image and model
image = cv2.imread("HOG Algorithm\people2.jpg")
cnn_detector = dlib.cnn_face_detection_model_v1("CNN Face Detection/mmod_human_face_detector.dat")


detections = cnn_detector(image, 1)


for face in detections:
    l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
    cv2.rectangle(image, (l,t), (r,b), (255,0,0), 3)


cv2.imshow("image", image)
cv2.waitKey()
