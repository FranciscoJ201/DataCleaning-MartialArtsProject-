import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (6,9)
#outline the dimensions of the checkerboard (0-6,0-9)
#type, max iterations, epsilon(accuracy threshold)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#vectors for 3d and 2d Points
threeDpoints = []
twoDPoints = []

#3D points real world coords
objectp3d = np.zeros((1,CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

prev_img_shape = None

#Taking path of individual stored images
#Take current directory saved jpgs

images = glob.glob('*.jpg')

for filename in images:
    image = cv2.imread(filename)
    #convert color to greyscale for faster processing
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Find the corners of the chess board and if they are found ret = true
    ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    #if desired number of corners is detected then refine the pixel coords nad display them on the images of the checkerboard
    if ret == True:
        threeDpoints.append(objectp3d)

        #Refining pixel coords for given 2d points

        corners2 = cv2.cornerSubPix(grayColor, corners, (11,11),(-1,-1), criteria)
        twoDPoints.append(corners2)
        #draw and display the corners 
        image = cv2.drawChessboardCorners(image,CHECKERBOARD,corners2,ret)

    cv2.imshow('img',image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = image.shape[:2]

#Perform the calibration by passing the value of found out 3d points and the corresponding pixel coordinates of the detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threeDpoints, twoDPoints, grayColor.shape[::-1], None, None)

print("Camera Matrix: ")
print(matrix)

print("\n Distorition coefficient: ")
print(distortion)

print("\n Rotation Vectors: ")
print(r_vecs)

print('\n Translation Vectors')
print(t_vecs)