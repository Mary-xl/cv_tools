import cv2
import numpy as np
import matplotlib as plt



def histogram_equalization(image):

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # only for 1 channel
    # convert the YUV image back to RGB format
    new_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # y: luminance(������), u&v: ɫ�ȱ��Ͷ�

    cv2.imshow("histogram equalization", np.hstack([image, new_image]))
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()
    return new_image