
import cv2
import numpy as np

def match_images(kpsA, kpsB, desA,desB, ratio=0.75):
    matcher=cv2.DescriptorMatcher_create("BruteForce")
    rawMatches=matcher.knnMatch(desA,desB,2)

    matches=[]
    for m in rawMatches:
        if m[0].distance <m[1].distance *ratio:
           matches.append([m[0].queryIdx,m[0].trainIdx])

    ptsA=[]
    ptsB=[]
    if len(matches)>4:
        for ms in matches:
            queryIdx=ms[0]
            ptsA.append(kpsA[queryIdx])
            trainIdx=ms[1]
            ptsB.append(kpsB[trainIdx])
    ptsA=np.array(ptsA)
    ptsB=np.array(ptsB)

    return (ptsA,ptsB)



