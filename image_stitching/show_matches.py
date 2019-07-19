



import cv2
import numpy as np

def drawMatches(img1, kps1, img2, kps2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    match_fig = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    match_fig[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
    match_fig[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    match_inlier_fig=match_fig.copy()

    for mat in matches:

        # Get the matching keypoints for each of the images
        # x - columns, y - rows
        (x1,y1) = (mat[0],mat[1])
        (x2,y2) = (mat[2],mat[3])

        # Draw a small circle at both co-ordinates
        cv2.circle(match_fig, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(match_fig, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        cv2.line(match_fig, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    for pair in inliers:

        # Draw a line in between the two inlier points
        (x1,y1) = (pair[0][0],pair[0][1])
        (x2,y2) = (pair[1][0],pair[1][1])

        cv2.circle(match_inlier_fig, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(match_inlier_fig, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        cv2.line(match_inlier_fig, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0), 1)

    return match_fig,match_inlier_fig