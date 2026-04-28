import cv2
import numpy as np

def tamper_score(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))

    diff = np.mean(cv2.absdiff(img1, img2)) / 255

    return float(diff)