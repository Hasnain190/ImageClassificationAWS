

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

WIDTH = 180
HEIGHT = 180
CHANNELS =1
TOP_CLASSES = 4

class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
def load_image(image_path):
    img = plt.imread(image_path)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    if len(img.shape) > 2: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/ 255
    img = np.expand_dims(img, axis=0)    
    return img

    
def predict(image_path):
    model = tf.keras.models.load_model('./my_model.h5')

    img = load_image(image_path)

    pred = model.predict(img)
    
    return pred




prediction = predict('./00000061_013.png')

# Get the top 4 predicted classes
top_4_classes = prediction.argsort()[0][-TOP_CLASSES:][::-1]

# Print the top 4 predicted classes
for class_index in top_4_classes:
    probable_disease = class_names[class_index]
    print(probable_disease)



