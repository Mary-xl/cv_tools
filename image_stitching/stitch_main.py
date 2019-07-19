# author: Mary Li
# date: July 2019
# part of the code is inspired by jhughes'code on Homography



import cv2
import numpy as np
from image_stitching.feature_detect import detect_features
from image_stitching.image_matching import match_images
from image_stitching.RANSAC_filter import RANSAC_filter
from image_stitching.show_matches import drawMatches


#by default stitch from left(img1) to right(img2)
def stitch_images(img1, img2):


    if len(img1.shape)>2:
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    if len(img2.shape)>2:
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


    (kps1, descriptors1) = detect_features(img1)
    (kps2, descriptors2) = detect_features(img2)

     #return matching pairs as candidate for RANSAC filtering
    (ptsA,ptsB)=match_images(kps1,kps2,descriptors1,descriptors2)
    matches=np.hstack([ptsA,ptsB])

    (h, maxInliers)=RANSAC_filter(ptsA,ptsB)
    matchFigure=drawMatches(img1, kps1, img2, kps2, matches, maxInliers)
    return matchFigure



if __name__=='__main__':

    img1=cv2.imread('/home/mary/AI_Computing/CV_DL/data/hallway1.jpg')
    img2=cv2.imread('/home/mary/AI_Computing/CV_DL/data/hallway2.jpg')

    matchFigure, matchRansacFigure=stitch_images(img1, img2)
    cv2.imwrite('/home/mary/AI_Computing/CV_DL/data/sift_matches.jpg', matchFigure)
    cv2.imwrite('/home/mary/AI_Computing/CV_DL/data/ransac_matches.jpg', matchRansacFigure)



