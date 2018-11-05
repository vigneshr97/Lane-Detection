#Advanced Lane Finding Project
#The goals / steps of this project are the following:
#Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#Apply a distortion correction to raw images.
#Use color transforms, gradients, etc., to create a thresholded binary image.
#Apply a perspective transform to rectify binary image ("birds-eye view").
#Detect lane pixels and fit to find the lane boundary.
#Determine the curvature of the lane and vehicle position with respect to center.
#Warp the detected lane boundaries back onto the original image.
#Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

import numpy as np 
import pickle
import cv2
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
import os

cal_images = glob.glob('camera_cal/calibration*.jpg')

def calibrate():
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
	objpoints = []
	imgpoints = []
	for idx, fname in enumerate(cal_images):
		image = cv2.imread(fname)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
		print(fname+' '+str(ret))
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)
			#cv2.drawChessboardCorners(img, (8,6), corners, ret)
	return objpoints, imgpoints

def undistort(img, objpoints, imgpoints):
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist

def hls_pipeline(img, s_thresh = (180, 255), sxthresh = (10, 100)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = 3)
	sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize = 3) 
	abs_sobelx = np.absolute(sobelx)
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	#abs_sobel_dir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))

    #sdirbinary = np.zeros_like(scaled_sobel)
    #sdirbinary[((abs_sobel_dir>=dir_thresh[0])&(abs_sobel_dir<=dir_thresh[1]))] = 1

	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sxthresh[0]) & (scaled_sobel <= sxthresh[1])] = 1

	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	#both combined
	combo = np.zeros_like(scaled_sobel)
	combo[(sxbinary==1)|(s_binary==1)] = 1
	combo *= 255
	# Stack each channel
	#color_binary = np.dstack((combo,combo,combo))*255
	#color_binary = np.dstack((sxbinary,s_binary), np.dot(sxbinary,s_binary), np.dot(sxbinary,s_binary))) * 255
	color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
	#cv2.imwrite('combo.jpg',combo)
	#cv2.imwrite('color.jpg',color_binary)
	return combo

def unwarp_image(img):
	img_size = (img.shape[1],img.shape[0])
	#src = np.float32([[img.shape[1]/2-55,img.shape[0]/2+100],[img.shape[1]/2+55,img.shape[0]/2+100],[(img.shape[1]*5/6)+60,img.shape[0]],[img.shape[1]/6-10,img.shape[0]]])
	src = np.float32([[img.shape[1]/2-60,img.shape[0]/2+90],[img.shape[1]/2+60,img.shape[0]/2+90],[(img.shape[1]*3/4)+140,img.shape[0]-20],[img.shape[1]/4-110,img.shape[0]-20]])
	dst = np.float32([[img.shape[1]/4,0],[img.shape[1]*3/4,0],[img.shape[1]*3/4,img.shape[0]],[img.shape[1]/4,img.shape[0]]])
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
	#warped_color = cv2.warpPerspective(undist, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
	#cv2.imwrite('warped.jpg',warped)
	#cv2.imwrite('warped_color.jpg',warped_color)
	#cv2.imwrite('original.jpg',img)
	return warped, M, Minv

def find_lane_pixels(img):
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
	out_img = np.dstack((img, img, img))
	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	nwindows = 9
	margin = 100
	minpix = 50
	window_height = np.int(img.shape[0]//nwindows)
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	leftx_current = leftx_base
	rightx_current = rightx_base
	left_lane_inds = []
	right_lane_inds = []

	for window in range(nwindows):
		win_y_low = img.shape[0] - (window+1)*window_height
		win_y_high = img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		#cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		#cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))    

	try:
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)
	except ValueError:
		pass

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	return leftx, lefty, rightx, righty, out_img

def fit_polynomial(img):
	leftx, lefty, rightx, righty, out_img = find_lane_pixels(img)
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	try:
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	except TypeError:
		# Avoids an error if `left` and `right_fit` are still none or incorrect
		print('The function failed to fit a line!')
		left_fitx = 1*ploty**2 + 1*ploty
		right_fitx = 1*ploty**2 + 1*ploty

	#storing all the points in the curve
	leftfitpt = []
	rightfitpt = []
	for i in range(len(ploty)):
		leftfitpt.append([left_fitx[i],ploty[i]])
		rightfitpt.append([right_fitx[i],ploty[i]])
	
	## Visualization ##
	# Colors in the left and right lane regions
	out_img[lefty, leftx] = [255, 0, 0]
	out_img[righty, rightx] = [0, 0, 255]

	# Plots the left and right polynomials on the lane lines
	#plt.plot(left_fitx, ploty, color='yellow')
	#plt.plot(right_fitx, ploty, color='yellow')
	leftfitpt = np.array([leftfitpt],np.int32)
	rightfitpt = np.array([rightfitpt],np.int32)
	leftfitpt.reshape((-1,1,2))
	rightfitpt.reshape((-1,1,2))
	out_img = cv2.polylines(out_img,[leftfitpt],False,(0,255,255),2)
	out_img = cv2.polylines(out_img,[rightfitpt],False,(0,255,255),2)
	return out_img, left_fit, right_fit

def search_around_poly(img, left_fit, right_fit):
	margin = 10
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Fit new polynomials
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	## Visualization ##
	# Create an image to draw on and an image to show the selection window
	out_img = np.dstack((img, img, img))*255
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# Plot the polynomial lines onto the image
	leftfitpt = []
	rightfitpt = []
	for i in range(len(ploty)):
		leftfitpt.append([left_fitx[i],ploty[i]])
		rightfitpt.append([right_fitx[i],ploty[i]])
	
	#plt.plot(left_fitx, ploty, color='yellow')
	#plt.plot(right_fitx, ploty, color='yellow')
	leftfitpt = np.array([leftfitpt],np.int32)
	rightfitpt = np.array([rightfitpt],np.int32)
	leftfitpt.reshape((-1,1,2))
	rightfitpt.reshape((-1,1,2))
	result = cv2.polylines(result,[leftfitpt],False,(0,255,255),2)
	result = cv2.polylines(result,[rightfitpt],False,(0,255,255),2)
	## End visualization steps ##
	return result, left_fit, right_fit

def convolution(img):
	window_width = 50 
	window_height = 80 # Break image into 9 vertical layers since image height is 720
	margin = 100 # How much to slide left and right for searching
	window_centroids = [] # Store the (left,right) window centroid positions per level
	window = np.ones(window_width) # Create our window template that we will use for convolutions
	l_sum = np.sum(img[int(3*img.shape[0]/4):,:int(img.shape[1]/2)], axis=0)
	l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
	r_sum = np.sum(img[int(3*img.shape[0]/4):,int(img.shape[1]/2):], axis=0)
	r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(img.shape[1]/2)

	window_centroids.append((l_center,r_center))

	for level in range(1,(int)(img.shape[0]/window_height)):
		image_layer = np.sum(img[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),:], axis=0)
		conv_signal = np.convolve(window, image_layer)
		offset = window_width/2
		l_min_index = int(max(l_center+offset-margin,0))
		l_max_index = int(min(l_center+offset+margin,img.shape[1]))
		l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
		r_min_index = int(max(r_center+offset-margin,0))
		r_max_index = int(min(r_center+offset+margin,img.shape[1]))
		r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
		window_centroids.append((l_center,r_center))

	if len(window_centroids) > 0:
		l_points = np.zeros_like(img)
		r_points = np.zeros_like(img)
		
		for level in range(0,len(window_centroids)):
			# Window_mask is a function to draw window areas
			l_mask = np.zeros_like(img)
			r_mask = np.zeros_like(img)
			l_mask[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),max(0,int(window_centroids[level][0]-window_width/2)):min(int(window_centroids[level][0]+window_width/2),img.shape[1])] = 1
			r_mask[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),max(0,int(window_centroids[level][1]-window_width/2)):min(int(window_centroids[level][1]+window_width/2),img.shape[1])] = 1
			# Add graphic points from window mask here to total pixels found 
			l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
			r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

		# Draw the results
		template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
		zero_channel = np.zeros_like(template) # create a zero color channel
		template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
		warpage= np.dstack((img, img, img))*255 # making the original road pixels 3 color channels
		output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 		# If no window centers found, just display orginal road image
	else:
		output = np.array(cv2.merge((img,img,img)),np.uint8)
	return output

def measure_curvature_pixels(img, left_fit, right_fit):
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	return left_curverad, right_curverad

def measure_curvature_real(img, left_fit, right_fit):
	ym_per_pix = 30/720
	xm_per_pix = 3.7/(img.shape[1]/2+250)
	ploty = np.linspace(0,img.shape[0]-1, img.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	return left_curverad, right_curverad

def normal_view_transform(img, undist, warped, left_fit, right_fit, Minv):
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	ploty = np.linspace(0,img.shape[0]-1, img.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

	pts = np.hstack((pts_left, pts_right))
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	return result

objpoints, imgpoints = calibrate()
i = 0
def pipeline(image):
	global i
	global left_fit
	global right_fit
	undist = undistort(image, objpoints, imgpoints)
	hls = hls_pipeline(undist)
	unwarped, perspective_M, Minv = unwarp_image(hls)
	if i == 0:
		out_img, left_fit, right_fit = fit_polynomial(unwarped)
	else:
		result, left_fit, right_fit = search_around_poly(unwarped, left_fit, right_fit)
	i+=1
    left_curverad, right_curverad = measure_curvature_real(image, left_fit, right_fit)
    final_output = normal_view_transform(image, undist, unwarped, left_fit, right_fit, Minv)
    left_fitx = left_fit[0]*(image.shape[0]-1)**2 + left_fit[1]*(image.shape[0]-1) + left_fit[2]
    right_fitx = right_fit[0]*(image.shape[0]-1)**2 + right_fit[1]*(image.shape[0]-1) + right_fit[2]
    distance = (image.shape[1]/2 - (left_fitx+right_fitx)/2)*3.7/(image.shape[1]/2+250)
    print(distance)
    if distance > 0:
        leftorright = 'right'
    else:
        leftorright = 'left'
        distance *= -1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_output, 'Radius of Curvature: '+str(round((left_curverad+right_curverad)/2, 2))+'m', (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(final_output, 'Position of the car: '+str(round(distance, 2))+'m '+leftorright+' from the centre', (230,100), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return final_output

# test_cap = cv2.VideoCapture('project_video.mp4')
# ret, frame = test_cap.read()
# test_cap.release()
cap = cv2.VideoCapture('project_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (720,1280))
objpoints, imgpoints = calibrate()
cv2.imwrite('output_images/calibrated_chess_board.jpg', undistort(cv2.imread('camera_cal/calibration2.jpg'), objpoints, imgpoints))

while cap.isOpened():
	ret, frame = cap.read()
	if ret == False:
		break
	final_output = pipeline(frame)
	cv2.imshow('frame', final_output)
	cv2.imwrite('output/final'+str(i)+'.jpg',final_output)
	out.write(final_output)
cap.release()
out.release()
cv2.destroyAllWindows()