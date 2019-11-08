import cv2
import numpy as np


def gamma_correction(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma=1.0/gamma
    a=[((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_image=cv2.LUT(image,table)

    cv2.putText(new_image, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("gamma correction", np.hstack([image, new_image]))
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()
    return new_image