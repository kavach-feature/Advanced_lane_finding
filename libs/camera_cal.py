import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
#%matplotlib inline



images= glob.glob('./camera_cal/calibration*.jpg')

def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist, mtx, dist

def compute_calibration_points():

    objpoints=[]
    imgpoints=[]
    objp=np.zeros((6*9,3), np.float32)
    objp[:,:2]= np.mgrid[0:9,0:6].T.reshape(-1,2)
	
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)	
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)        
    return imgpoints, objpoints

def render_original_undistorted_image(image1, image2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title("Original Image", fontsize=50)
    ax2.imshow(image2)
    ax2.set_title("Distortion Corrected image", fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()



def get_undistorted_image(orig_image):
    imgpoints, objpoints = compute_calibration_points()
    undist_road_image, mtx, dist_coefficients = cal_undistort(orig_image, objpoints, imgpoints)
    return undist_road_image


"""   
imgpoints, objpoints = compute_calibration_points()
img = mpimg.imread('./camera_cal/calibration4.jpg')
undist, mtx, dist_coefficients = cal_undistort(img, objpoints, imgpoints)
#render_original_undistorted_image(img, undist)

image_path = './test_images/test5.jpg'
image = mpimg.imread(image_path)
undist_road_image, mtx, dist_coefficients = cal_undistort(image, objpoints, imgpoints)
render_original_undistorted_image(image,undist_road_image)"""