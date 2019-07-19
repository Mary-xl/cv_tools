
import cv2
import numpy as np

def detect_features(img):

    if len(img.shape)>2:
       img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #create SIFT detector
    detector=cv2.xfeatures2d.SIFT_create()
    (kps, descriptors) = detector.detectAndCompute(img, None)
    #transfer kps from objects to numpy arrays
    kps=np.float32([kp.pt for kp in kps])
    return (kps, descriptors)



