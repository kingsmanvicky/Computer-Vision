import os
import cv2
import dlib
from PIL import Image   
import numpy as np
from sklearn.metrics import accuracy_score


face_detector = dlib.get_frontal_face_detector()    
points_detector = dlib.shape_predictor("Facial Points and descriptors detection\Shape_predictor_trained.dat")
face_descriptor_extractor = dlib.face_recognition_model_v1("Facial Points and descriptors detection\Facial_descriptors_trained.dat")


index = {}
idx = 0
face_descriptors = None


paths = [os.path.join("Facial Points and descriptors detection\yalefaces\Train", f) for f in os.listdir("Facial Points and descriptors detection\yalefaces\Train")]
for path in paths:
    image = Image.open(path).convert("RGB")
    image_np = np.array(image, 'uint8')

    face_detection  = face_detector(image_np, 1)
    for face in face_detection:
        l, t, r, b, = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(image_np, (l,t), (r,b), (0,255,0), 2)

        points = points_detector(image_np, face)
        for point in points.parts():
            cv2.circle(image_np, (point.x, point.y),2,(255,0,0), 1)

        face_descriptor = face_descriptor_extractor.compute_face_descriptor(image_np, points)
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor)
        face_descriptor = face_descriptor[np.newaxis, :]
        

        if face_descriptors is None:
            face_descriptors = face_descriptor
        else:
            face_descriptors = np.concatenate((face_descriptors, face_descriptor))
        
        index[idx] = path
        idx += 1

threshold = 0.5
predictions = []
expected_outputs = []


paths = [os.path.join("Facial Points and descriptors detection\yalefaces\Test", f) for f in os.listdir("Facial Points and descriptors detection\yalefaces\Test")]
for path in paths:
    image = Image.open(path).convert("RGB")
    image_np = np.array(image, 'uint8')
    face_detection = face_detector(image_np, 1)
    for face in face_detection:
        points = points_detector(image_np, face)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(image_np, points)
        face_descriptor = [f for f in face_descriptor]
        face_descriptor = np.asarray(face_descriptor)
        face_descriptor = face_descriptor[np.newaxis, :]

        distances = np.linalg.norm(face_descriptor - face_descriptors, axis = 1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        if min_distance <= threshold:
            name_pred = int(os.path.split(index[min_index])[1].split(".")[0].replace('subject',''))
        else:
            name_pred = "Not Identified"
        
        name_real = int(os.path.split(path)[1].split(".")[0].replace('subject',''))

        predictions.append(name_pred)
        expected_outputs.append(name_real)

    #     cv2.putText(image_np, f"Pred: {name_pred}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    #     cv2.putText(image_np, f"Exp: {name_real}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

    # cv2.imshow("Image", image_np)
    # cv2.waitKey()


acc = accuracy_score(expected_outputs, predictions)
print(acc)


