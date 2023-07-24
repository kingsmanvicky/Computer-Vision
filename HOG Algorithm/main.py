import dlib
import cv2


image = cv2.imread("HOG Algorithm\people2.jpg")
face_detector_Hog = dlib.get_frontal_face_detector()
detections = face_detector_Hog(image, 1)


for face in detections:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l,t), (r,b), (255,0,0), 5)


cv2.imshow("Image", image)
cv2.waitKey()
