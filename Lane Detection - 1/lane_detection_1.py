import numpy as np 
import pickle
import cv2
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
import os

#cap = cv2.VideoCapture('challenge.mp4')
#cap = cv2.VideoCapture('solidWhiteRight.mp4')
cap = cv2.VideoCapture('test_videos/solidYellowLeft.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 25.0, (540,960))
i = 1
kernel_size = 5
low_threshold = 50
high_threshold = 150
rho = 4
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 20
j = 0
while(cap.isOpened()):
	ret, frame = cap.read(0)
	if ret == False:
		break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	line_image = np.copy(frame)*0
	mask = np.zeros_like(edges)   
	ignore_mask_color = 255   
	imshape = frame.shape
	vertices = np.array([[(0,imshape[0]),(470,320),(490,320),(imshape[1],imshape[0])]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
	slopes = []
	for line in lines:
		for x1,y1,x2,y2 in line:
			slopes.append((y2-y1)/(x2-x1));
	slopes.sort()
	new_lines = []
	for i in slopes:
		if ():
			pass
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),5)
	color_edges = np.dstack((masked_edges, masked_edges, masked_edges))
	combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
	combo_img = cv2.addWeighted(frame, 1, line_image, 1, 0)
	cv2.imwrite(('output/Frame'+str(j)+'.jpg'),combo_img)
	j += 1
	# if ret == True:
	# 	out.write(combo_img)
	# 	cv2.imshow('Frame',combo_img)
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		break
	# else:
	# 	break
cap.release()
#out.release()
#cv2.destroyAllWindows()
