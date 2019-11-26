import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


objpoints=[]
imgpoints=[]
objp=np.zeros((6*9,3), np.float32)
objp[:,:2]= np.mgrid[0:9,0:6].T.reshape(-1,2)



# Defines a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh=(0,255)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply cv2.Sobel()
    if orient == 'x':
        sobel_orient= cv2.Sobel(gray,cv2.CV_64F,1,0)
    elif orient == 'y':
    	sobel_orient= cv2.Sobel(gray,cv2.CV_64F,0,1)
    # Take the absolute value of the output from cv2.Sobel()
    abs_sobel = np.absolute(sobel_orient)
    # Scale the result to an 8-bit range (0-255)
    scaled_sobel= np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output= np.zeros_like(scaled_sobel)

    # Apply lower and upper thresholds
    binary_output [(scaled_sobel >=thresh[0])&(scaled_sobel<=thresh[1])]=1
    # Create binary_output
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Grayscale
   # Apply the following steps to img
    # 1) Convert to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_orient_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_orient_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)

    # 3) Calculate the magnitude 
    abs_sobel= np.sqrt(sobel_orient_x**2 + sobel_orient_y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel=np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    binary_output=np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel>=mag_thresh[0]) & (scaled_sobel<=mag_thresh[1])]=1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output
    
    
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_orient_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_orient_y= cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx=np.absolute(sobel_orient_x)
    abs_sobely=np.absolute(sobel_orient_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_gradient = np.arctan2(abs_sobely,abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output= np.zeros_like(dir_gradient)
    binary_output[(dir_gradient>= thresh[0])& (dir_gradient<= thresh[1])]=1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls_image= cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    S=hls_image[:,:,2]
    # 2) Apply a threshold to the S channel
    threshold=(190,255)
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    binary_output= np.zeros_like(S)
    binary_output[(S>=threshold[0])&(S<=threshold[1])]=1
    return binary_output


def  birds_eye_view(image):
    img_size= (image.shape[1],image.shape[0])

    src = np.float32(
        [[490,480],
        [810,480],
        [1250,720],
        [140,720]])

    dst = np.float32(
        [[0,0],
        [1280,0],
        [1250,720],
        [140,720]])


    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped_image = cv2.warpPerspective(image,M, img_size,flags=cv2.INTER_NEAREST)
    return warped_image, Minv

def create_threshold_binary_image(image,ksize=3):
    ksize=3
    gradx=abs_sobel_thresh(image,orient='x',sobel_kernel=ksize,thresh=(20,100))    
    grady=abs_sobel_thresh(image,orient='y',sobel_kernel=ksize,thresh=(20,100))
    #Applying magnitude threshold
    mag_binary = mag_thresh(image,sobel_kernel=ksize,mag_thresh=(30,100))
    #Applying threshold where the vertical direction of the gradient is met
    dir_output= dir_threshold(image,sobel_kernel=ksize,thresh=(0.7,1.3))
    #Applying HLS color space threshold
    hls_output = hls_select(image)
    
    """Creating a binary image where only non-zero pixels meeting absolute Sobelx threshold, 
    magnitude based threshold (for Sobelx and Sobel y) and direction gradient meets the criteria"""
    combined = np.zeros_like(dir_output)
    combined [(gradx==1)&(grady ==1)| (mag_binary ==1)& (dir_output ==1)] =1
    combined_with_color_threshold=np.zeros_like(combined)
    combined_with_color_threshold [(hls_output==1)|(combined==1)]=1
    #combined_with_color_threshold [(combined==1)]=1
    
    return combined_with_color_threshold




def render_original_combined_transforms_image(image1, image2,image2_title ="Processed Image"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 9))
    f.tight_layout()
    ax1.imshow(image1,cmap='gray')
    ax1.set_title("Original Image", fontsize=10)
    ax2.imshow(image2,cmap='gray')
    ax2.set_title(image2_title, fontsize=10)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.5)
    plt.show()

def render_original_plotted_warped_image(image1, image2,image2_title ="Processed Image"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 9))
    f.tight_layout()
    ax1.imshow(image1,cmap='gray')
    """ax1.plot([695,1100],[450,700], color='r', linewidth="6")
    ax1.plot([1100,225],[700,700], color='r', linewidth="6")
    ax1.plot([225,600],[700,470], color='r', linewidth="6")
    ax1.plot([600,695],[470,450], color='r', linewidth="6")"""
    ax1.set_title("Original Image", fontsize=10)
    ax2.imshow(image2,cmap='gray')
    """ax2.plot([900,900],[0,700], color='r', linewidth="6")
    ax2.plot([900,255],[700,700], color='r', linewidth="6")
    ax2.plot([255,255],[700,0], color='r', linewidth="6")
    ax2.plot([255,900],[0,0], color='r', linewidth="6")"""
    
    ax2.set_title(image2_title, fontsize=10)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.5)
    plt.show()



image_path = './test_images/straight_lines1.jpg'
image = mpimg.imread(image_path)
combined_image= create_threshold_binary_image(image,3)
render_original_combined_transforms_image(image,combined_image,"Thresholded Binary Image")
"""
image_path = './test_images/test6.jpg'
image = mpimg.imread(image_path)
combined_image= create_threshold_binary_image(image,3)
warped_image= birds_eye_view(combined_image)
render_original_plotted_warped_image(combined_image,warped_image,"Birds Eye view Image")"""