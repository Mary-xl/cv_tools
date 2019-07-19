
import cv2
import numpy as np
import random
from image_stitching.get_homography import calculateHomography

def RANSAC_filter(ptsA, ptsB):

#randomly pick 4 pairs

    #number of pairs before filtering
  num=ptsA.shape[0]

  maxInliers=[]
  for it in range (0,200):
      picked_four=[]
      for i in range(0,4):
          idx=random.randrange(0,num)
          pair=(ptsA[idx], ptsB[idx])
          picked_four.append(pair)
      #put the 4 pairs into homography calculation
      h=calculateHomography(picked_four)

      inliers = []
      #pick one pair
      for j in range (0,num):
          pair=(ptsA[j], ptsB[j])
          error=difference(pair,h)
          if error <5:
             inliers.append(pair)

      if len(inliers) > len(maxInliers):
          maxInliers = inliers
          finalH = h

      print ("number of pairs: ",num, "max inliers: ", len(maxInliers) )

  print ('ok')
  return (h, maxInliers)



def difference(pair,h):


        pta=pair[0]
        ptb=pair[1]

        p1 = np.transpose(np.matrix([pta[0], pta[1], 1]))
        est_p2 = np.dot(h, p1)
        est_p2 = (1 / est_p2.item(2)) * est_p2

        p2 = np.transpose(np.matrix([ptb[0], ptb[1], 1]))
        diff=p2 - est_p2

        error=np.linalg.norm(diff)
        return  error






