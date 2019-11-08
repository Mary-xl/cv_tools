import cv2


#imreadChannels
#when read in any image (including a grayscale image), OpenCV will by default read in 3 channels and if a grayscale then set all 3 channels to the same value
img=cv2.imread("/home/mary/AI_Computing/CV_DL/data/hallway1.jpg")
print (img.shape) #(570, 855, 3)

grayImg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print (grayImg.shape) #(570, 855)
cv2.imwrite("/home/mary/AI_Computing/CV_DL/data/hallway1_g.jpg", grayImg)

img2=cv2.imread("/home/mary/AI_Computing/CV_DL/data/hallway1_g.jpg")
print (img2.shape) #(570, 855, 3) [[[170 170 170][170 170 170]...]]]
img3=cv2.imread("/home/mary/AI_Computing/CV_DL/data/hallway1_g.jpg",0)
print (img3.shape) #

