#copyright: Mary Li, July 2019


import cv2
import numpy as np


def shift_color(img,s=0):
    B,G,R=cv2.split(img)

    channels=[B,G,R]
    for X in channels:
        if s==0:
            pass

        #set up upper bound for the shift
        elif s>0:
           lim_up=255-s
           X[X > lim_up] = 255
           X[X<lim_up]=(s+X[X<lim_up]).astype(img.dtype)

        #set up lower bound for the shift
        elif s<0:
           lim_low=0-s
           X[X<lim_low]=0
           X[X>lim_low]=(s+X[X>lim_low]).astype(img.dtype)

    #merge the new new values from 3 channels to form a new shifted image
    new_img=cv2.merge([B,G,R,])
    cv2.putText(new_img, "shift={}".format(s), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("random shift", np.hstack([img, new_img]))
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()
    return new_img