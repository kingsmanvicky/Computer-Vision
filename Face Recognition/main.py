from PIL import Image
import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn 
import matplotlib.pyplot as plt

#Preprocessing the images and getting the subject ids
def get_image_data():
    paths = [os.path.join("Face Recognition\yalefaces\Train",f) for f in os.listdir("Face Recognition\yalefaces\Train")]
    faces = []
    ids = []
    for path in paths:
        image = Image.open(path).convert('L')
        image_np = np.array(image, 'uint8')
        id = int(os.path.split(path)[1].split('.')[0].replace('subject',''))
        ids.append(id)
        faces.append(image_np)

    return faces, np.array(ids)


faces, ids = get_image_data()

#Training the LBPH face recognizer 
# lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius = 4, neighbors = 14, grid_x = 9, grid_y = 9)
# lbph_classifier.train(faces, ids)
# lbph_classifier.write("LBPH_Classifier.yml")
# exit()

#Recognizing faces
lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read("Face Recognition\LBPH_Classifier.yml")


############### Classifying the images after training ###############

#Pre-Processing the test image
test_image = "Face Recognition\yalefaces\Train\subject10.wink.gif"
image = Image.open(test_image).convert("L")
image_np = np.array(image, 'uint8')


#Evaluating the Algorithm
paths = [os.path.join('Face Recognition\yalefaces\Test', f) for f in os.listdir("Face Recognition\yalefaces\Test")]
predictions = []
expected_outputs =[]

for path in paths:
    image = Image.open(path).convert("L")
    image_np = np.array(image, 'uint8')
    prediction,_ = lbph_face_classifier.predict(image_np)
    expected_output  = int(int(os.path.split(path)[1].split('.')[0].replace('subject','')))

    predictions.append(prediction)
    expected_outputs.append(expected_output)

predictions = np.array(predictions) 
expected_outputs = np.array(expected_outputs)

score = f"Score : {accuracy_score(expected_outputs, predictions)*100}%"
cm = confusion_matrix(expected_outputs, predictions)
cm_graph = seaborn.heatmap(cm, annot =True)

print(score)
plt.show()

