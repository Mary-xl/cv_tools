
import cv2
import numpy

def image_crop(image):

    #crop rectangles
    start_x = int(input("please specify crop starting point x:"))
    start_y = int(input("please specify crop starting point y:"))

    end_x = int(input("please specify crop ending point x:"))
    end_y = int(input("please specify crop ending point y:"))

    new_image=image[start_y:end_y,start_x:end_x]
    cv2.imshow("cropped image",new_image)
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()
    return new_image