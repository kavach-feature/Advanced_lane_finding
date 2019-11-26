import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from libs import dir_sobel_color_persp_warp_func
from libs import camera_cal
from libs import sliding_window_histogram

from line import Line
from moviepy.editor import VideoFileClip


frame_size= 5
left_line = Line(n=frame_size)
right_line = Line(n=frame_size)
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
left_lane_inds, right_lane_inds = None, None  # for calculating curvature

detected = False


def pipeline_rendering_line_fit(image):
	
	global detected,left_curve,right_curve,left_lane_inds,right_lane_inds

	combined_image= dir_sobel_color_persp_warp_func.create_threshold_binary_image(image,3)
	binary_warped_image, reverse_M= dir_sobel_color_persp_warp_func.birds_eye_view(combined_image)

	if not detected:

		ret, out_img, left_fitx, right_fitx, ploty = sliding_window_histogram.fit_polynomial(binary_warped_image)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		#Update the running average

		left_fit = left_line.update_fit(left_fit)
		right_fit = right_line.update_fit(right_fit)

		detected = True
	else:
		left_fit=left_line.obtain_fit()
		right_fit=right_line.obtain_fit()

		ret =sliding_window_histogram.optimized_fit_polynomial(binary_warped_image,left_fit,right_fit)
		
		"""		
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']
		right_fit = ret['right_fit']
		left_fit = ret['left_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']"""
		
		if ret is not None:

			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']

			left_fit = left_line.update_fit(left_fit)
			right_fit = right_line.update_fit(right_fit)
		else:
			detected = False

	result = sliding_window_histogram.render_radius_offset_orig_image(image,reverse_M, binary_warped_image)

	return result

def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(pipeline_rendering_line_fit)
	annotated_video.write_videofile(output_file, audio=False)


annotate_video('harder_challenge_video.mp4', 'harder_challenge_out.mp4')
image_path = './test_images/straight_lines1.jpg'
image = mpimg.imread(image_path)
#result = pipeline_rendering_line_fit(image)






























