import cv2
import numpy as np

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
    return img