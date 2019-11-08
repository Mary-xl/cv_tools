import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def similarity_transform(image, center, dim, angle=0.0, scale=1.0, clip=True):
    #similarity transform includes rotation+translation+scaling
    #it retains parallel and angles
    M = cv2.getRotationMatrix2D(center, angle, scale)
    (w,h,d)=image.shape
    if clip==True:
      new_image=cv2.warpAffine(image, M, dim)
    elif clip==False:
      cos=np.abs(M[0,0])
      sin=np.abs(M[0,1])
      nw=w*cos+h*sin
      nh=w*sin+h*cos
      M[0,2]+=(nw/2)-center[0]
      M[1,2]+=(nh/2)-center[1]
      new_image = cv2.warpAffine(image, M, (int(nw),int(nh)))
    return  new_image

def affine_transform(image, pts1,pts2):
    #retians parallel, need 3 pairs of points to solve a 2*3 M matrix
    M=cv2.getAffineTransform(pts1,pts2)
    (w,h,d)=image.shape
    new_image=cv2.warpAffine(image, M,(w,h))
    return new_image

def perspective_transform(image, pts1,pts2):
    #retians straight lines, need 4 pairs of points to solve a 3*3 (8 dof) matrix
    (w, h, d) = image.shape
    M=cv2.getPerspectiveTransform(pts1,pts2)
    new_image=cv2.warpPerspective(image,M,(2*w,2*h))
    return new_image

def transform_img(img,center,img_dim):
    type = int(input("please specify transform type: 0 for similarity transform, 1 for affine transform, 2 for perspective transform: "))

    if type==0:
       center_x = input("please specify translation center x, using default hit enter: ")
       center_y = input("please specify translation center y, using default hit enter: ")
       if center_x=="":
           center_x=center[0]
       if center_y == "":
           center_y = center[1]
       center = (int(center_x), int(center_y))
       print(center)

       dim=img_dim

       angle = input("please specify image rotation angle, using default hit enter: ")
       if angle=="":
           angle=0
       angle=float(angle)

       scale =input("please specify image scaling factor, default hit enter: ")
       if scale=="":
           scale=1
       scale=float(scale)
       clip_flag=int(input("please specify if the image will be clipped, by default clipped input 1, if not input 0: "))

       if  clip_flag==1:
           clip=True
       else:
           clip=False
       new_image=similarity_transform(img, center, dim, angle, scale, clip)
       cv2.imshow("original", img)
       cv2.imshow("similarity transform", new_image)


    if type==1:

        option=input ("please provide 3 pairs of corresponding points, using default values as example input 0: ")
        if int(option)==0:
            w, h, d = img.shape
            pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
            pts2 = np.float32([[w * 0.2, h * 0.1], [w * 0.9, h * 0.2], [w * 0.1, h * 0.9]])

        new_image=affine_transform(img,pts1,pts2)
        cv2.imshow("original", img)
        cv2.imshow("affine transform", new_image)

        #using matrix transformation to verify
        new_matrix = dotProduct(img, pts1, pts2, type)



    if type==2:
        option=input ("please provide 4 pairs of corresponding points, using default values as example input 0: ")
        if int(option)==0:
            # warp:
            random_margin = 60
            width=img.shape[0]
            height=img.shape[1]
            x1 = random.randint(-random_margin, random_margin)
            y1 = random.randint(-random_margin, random_margin)
            x2 = random.randint(width - random_margin - 1, width - 1)
            y2 = random.randint(-random_margin, random_margin)
            x3 = random.randint(width - random_margin - 1, width - 1)
            y3 = random.randint(height - random_margin - 1, height - 1)
            x4 = random.randint(-random_margin, random_margin)
            y4 = random.randint(height - random_margin - 1, height - 1)

            dx1 = random.randint(-random_margin, random_margin)
            dy1 = random.randint(-random_margin, random_margin)
            dx2 = random.randint(width - random_margin - 1, width - 1)
            dy2 = random.randint(-random_margin, random_margin)
            dx3 = random.randint(width - random_margin - 1, width - 1)
            dy3 = random.randint(height - random_margin - 1, height - 1)
            dx4 = random.randint(-random_margin, random_margin)
            dy4 = random.randint(height - random_margin - 1, height - 1)

            pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])

        new_image=perspective_transform(img,pts1,pts2)
        cv2.imshow("original", img)
        cv2.imshow("perspective transform", new_image)

        #using matrix transformation to verify
        new_matrix = dotProduct(img, pts1, pts2, type)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    return  new_image

def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

def dotProduct(img, pts1, pts2, type):

        if type==1:
           M = cv2.getAffineTransform(pts1, pts2)
           one = np.ones((3, 1))
        if type==2:
           M = cv2.getPerspectiveTransform(pts1, pts2)
           one = np.ones((4, 1))
        pts1_add = np.append(pts1,one,axis=1)
        #transform_pts1=np.transpose(np.dot(M,np.transpose(pts1_add)))
        transform_pts1 = np.dot(pts1_add,np.transpose(M))
        if transform_pts1.all()==pts2.all():
            print ("M matrix confirmed")

        for idx, point in enumerate (pts1):
            xi_0,yi_0=point
            xi_1,yi_1=pts1[(idx+1)%len(pts1)]
            plt.plot([xi_0,xi_1],[yi_0,yi_1], color='yellow')

        for idx, point in enumerate (pts2):
            xi_0,yi_0=point
            xi_1,yi_1=pts2[(idx+1)%len(pts2)]
            plt.plot([xi_0,xi_1],[yi_0,yi_1],color='green')

        ax = plt.gca()  # get the current axis
        ax.xaxis.set_ticks_position('top')  # put x axis to the top
        ax.invert_yaxis() # invert y axis to the opposite direction so as be consistent as image coordinate system
        plt.show()
        cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)





