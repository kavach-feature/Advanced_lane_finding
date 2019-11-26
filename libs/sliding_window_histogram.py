import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from libs import dir_sobel_color_persp_warp_func
from libs import camera_cal
# Load our image
#binary_warped = mpimg.imread('warped_example.jpg')

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin from the point where the peak pixel value has been detected
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current-margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)
        &(nonzerox<win_xleft_high)&(nonzerox>=win_xleft_low)).nonzero()[0]
        good_right_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)
    	&(nonzerox<win_xright_high)&(nonzerox>=win_xright_low)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds)>minpix:
        	leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)>minpix:
        	rightx_current=np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # Return a dict of relevant variables
    ret = {}
    ret['left_fit'] = left_lane_inds
    ret['right_fit'] = right_lane_inds
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['out_img'] = out_img
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds


    return ret,leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    ret, leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='red')


    return ret,out_img, left_fitx, right_fitx, ploty


def optimized_fit_polynomial(binary_warped,left_fit,right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If we don't find enough relevant points, return all None (this means error)
    min_inds = 10
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        plt.imshow(binary_warped)
        plt.show()
        return None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Return a dict of relevant variables
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    print (ret['left_fit'])

    return ret




def compute_radius(binary_warped):
    ym_per_pix = 30/720 
    xm_per_pix= 3.7/660
    y_eval= 700

    ret, leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    left_y1 = (2*left_fit[0]*y_eval + left_fit[1])*xm_per_pix/ym_per_pix
    left_y2 = 2*left_fit[0]*xm_per_pix/(ym_per_pix * ym_per_pix)
    left_rad_curvature = ((1+left_y1*left_y2)**(1.5))/np.absolute(left_y2)

    right_y1 = (2*right_fit[0]*y_eval + right_fit[1])*xm_per_pix/ym_per_pix
    right_y2 =2*right_fit[0]*xm_per_pix/(ym_per_pix * ym_per_pix)
    right_rad_curvature = ((1+right_y1*right_y2)**(1.5))/np.absolute(right_y2)

    radius= left_rad_curvature 
    
    #print("Radius of Curvature: %f" % left_rad_curvature)
    return radius


def compute_offset(binary_warped):

    ym_per_pix = 30/720 
    xm_per_pix= 3.7/650
    y_eval= 700
    midx = 650

    ret, leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
    x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
    position_from_center = ((x_left_pix + x_right_pix)/2 - midx) * xm_per_pix
    if position_from_center < 0:
        text = 'left'
    else:
        text = 'right'

    #print("Distance of the car is %.2f" % position_from_center)
    #print("from" , text )  
    return text, position_from_center
    

def render_radius_offset_orig_image(orig_image,reverse_M, binary_warped):

    radius  = compute_radius(binary_warped)
    text, offset = compute_offset(binary_warped)      

    ret, out_img, left_fitx, right_fitx, ploty= fit_polynomial(binary_warped)

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    undistorted_img= camera_cal.get_undistorted_image(orig_image)

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, reverse_M, (undistorted_img.shape[1], undistorted_img.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

    cv2.putText(result,'Radius of Curvature: %.2fm' % radius ,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    if offset < 0:
        text= 'left'
    else:
        text='right'

    cv2.putText(result,'Car is traveling %.2fm %s of Center' % (np.absolute(offset), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    #plt.imshow(result)
    #plt.show()
    return result

def render_line_polynomial_orig_image(image1, image2,image2_title):
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    f.tight_layout()
    ax1.imshow(image1,cmap='gray')
    ax1.plot([695,1100],[450,700], color='r', linewidth="6")
    ax1.plot([1100,225],[700,700], color='r', linewidth="6")
    ax1.plot([225,600],[700,470], color='r', linewidth="6")
    ax1.plot([600,695],[470,450], color='r', linewidth="6")
    ax1.set_title("Original image",fontsize=8)

    
    ret, out_img, left_fitx, right_fitx, ploty= fit_polynomial(image2)
    ax2.imshow(image2,cmap='gray')
    ax2.plot([900,900],[0,700], color='r', linewidth="6")
    ax2.plot([900,255],[700,700], color='r', linewidth="6")
    ax2.plot([255,255],[700,0], color='r', linewidth="6")
    ax2.plot([255,900],[0,0], color='r', linewidth="6")
    ax2.set_title("warped_Image", fontsize=8)

    
    ax3.imshow(out_img,cmap='gray')
    ax3.plot(left_fitx, ploty, color='yellow')
    ax3.plot(right_fitx, ploty, color='red')
    ax3.set_title(image2_title, fontsize=8)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.5)
    plt.show()


"""
image_path = './test_images/straight_lines2.jpg'
image = mpimg.imread(image_path)
combined_image= dir_sobel_color_persp_warp_func.create_threshold_binary_image(image,3)
binary_warped_image, reverse_M= dir_sobel_color_persp_warp_func.birds_eye_view(combined_image)
result = render_radius_offset_orig_image(image,reverse_M, binary_warped_image)

#render_line_polynomial_orig_image(image,binary_warped_image,"Detected lanes")"""
