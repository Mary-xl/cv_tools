
# author: Mary Li
# date: July 2019

import cv2
import numpy as np
import os

def median_filter(img, kernel, paddingway, dataFolder):

    #convert RGB to gray
    if len(img.shape)==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #padding image
    H,W=img.shape
    new_img=img.copy()
    m, n = kernel.shape
    pad_img=padding(img,kernel,paddingway)

    temp_img=pad_img.copy()
    H_n,W_n=temp_img.shape

    #median filtering
    for i in range(0,W):
        for j in range(0,H):
            roi_img=temp_img[j:j+n,i:i+m]
            roi_temp=roi_img.ravel()
            median_temp=np.median(roi_temp)
            new_img[j,i]=median_temp

    cv2.imshow("filtered image",np.hstack([img,new_img]))
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()

    if int(paddingway)==0:
       output_path = os.path.join(dataFolder, 'median_filerted_zero.png')
       cv2.imwrite(output_path, new_img)
    if int(paddingway)==1:
       output_path = os.path.join(dataFolder, 'median_filerted_replica.png')
       cv2.imwrite(output_path, new_img)
    return new_img

def padding(img, kernel,paddingway):

    H,W=img.shape
    n,m=kernel.shape
    pad_x=int(m/2)
    pad_y=int(n/2)
    new_shape=[W+2*pad_x, H+2*pad_y]


    if int(paddingway)==0:
       pad_img=np.zeros((H+2*pad_y, W+2*pad_x), dtype=np.uint8)
       pad_img[pad_y:pad_y+H,pad_x:pad_x+W]=img


    elif int(paddingway)==1:
       pad_img=np.zeros((H+2*pad_y, W+2*pad_x), dtype=np.uint8)
       pad_img[pad_y:pad_y + H, pad_x:pad_x + W] = img
       for cidx in range(pad_x):
         pad_img[pad_y:pad_y+H,cidx]=img[:,0]
         pad_img[pad_y:pad_y+H, pad_x+W+cidx]=img[:,W-1]
       for ridy in range(pad_y):
         pad_img[ridy,pad_x:pad_x+W]=img[0,:]
         pad_img[ridy+H+pad_y,pad_x:pad_x+W]=img[H-1,:]
       print(pad_img.shape)


    cv2.imshow("padded image", pad_img)
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()
    return pad_img