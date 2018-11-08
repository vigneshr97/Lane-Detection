import csv
import cv2
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import random
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from scipy import misc
from scipy.ndimage import rotate
from skimage import transform
from skimage.transform import warp, SimilarityTransform, AffineTransform
from skimage import exposure


def apply_rotation(img):
    rotated = 255*transform.rotate(img, angle=np.random.uniform(-2, 2), mode='edge')
    rotated = rotated.astype(np.uint8)
    return rotated.astype(np.uint8)

def apply_translation(img):
    translated = 255*warp(img, transform.SimilarityTransform(translation=(np.random.uniform(-5, 5), np.random.uniform(-5, 5))),mode='edge')
    translated = translated.astype(np.uint8)
    return translated.astype(np.uint8)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE()
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    c = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return c.astype(np.uint8)

def modify_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand = 0.5+random.random()/2
    hsv[:,:,2] = hsv[:,:,2] * rand
    ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return ret.astype(np.uint8)

def gaussianblur(img):
    ret = cv2.GaussianBlur(img,(5,5),0)
    return ret.astype(np.uint8)

def medianblur(img):
    ret = cv2.medianBlur(img,5)
    return ret.astype(np.uint8)

lines = []
#data_file = 'orig_data/driving_log.csv'
data_file = 'data/driving_log.csv'
with open(data_file, 'r') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		lines.append(line)

data_file = 'data/driving_log_old.csv'
with open(data_file, 'r') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		lines.append(line)


random.shuffle(lines)
images = []
measurements = []
i = 0
for line in lines:
	if i > 10:
		if (((measurements[-1]==0)|(measurements[-2]==0)|(measurements[-3]==0)|(measurements[-4]==0)|(measurements[-5]==0)|(measurements[-6]==0)|(measurements[-7]==0)|(measurements[-8]==0)|(measurements[-9]==0)|(measurements[-10]==0))&(float(line[3])==0)):
			continue

	if i%500 == 0:
		print(i)
	i += 1
	#image = cv2.imread('orig_data/'+line[0])
	image = mpimg.imread(line[0])
	measurement = float(line[3])
	images.append(image)
	measurements.append(measurement)
	
	clahe_image = apply_clahe(image)
	brightness_image = modify_brightness(image)
	# trans_image = apply_translation(image)
	# gauss_image = gaussianblur(image)
	# med_image = medianblur(image)
	#images.append(trans_image)
	images.append(brightness_image)
	images.append(clahe_image)
	# images.append(gauss_image)
	# images.append(med_image)

	measurements.append(measurement)
	measurements.append(measurement)
	# measurements.append(measurement)
	# measurements.append(measurement)
	# measurements.append(measurement)

	# image1 = cv2.imread('orig_data/'+line[1][1:])
	# image1 = mpimg.imread(line[1])
	# measurement1 = float(line[3]) + 0.2
	# images.append(image1)
	# measurements.append(measurement1)

	
	# image2 = cv2.imread('orig_data/'+line[2][1:])
	# image2 = mpimg.imread(line[2])
	# measurement2 = float(line[3]) - 0.2
	# images.append(image2)
	# measurements.append(measurement2)
	
	image_flipped = np.fliplr(image)
	measurement_flipped = -measurement
	images.append(image_flipped)
	measurements.append(measurement_flipped)

	clahe_image_flipped = apply_clahe(image_flipped)
	brightness_image_flipped = modify_brightness(image_flipped)
	# trans_image_flipped = apply_translation(image_flipped)
	# gauss_image_flipped = gaussianblur(image_flipped)
	# med_image_flipped = medianblur(image_flipped)
	images.append(clahe_image_flipped)
	images.append(brightness_image_flipped)
	# images.append(gauss_image_flipped)
	# images.append(med_image_flipped)
	#images.append(trans_image_flipped)
	measurements.append(measurement_flipped)
	measurements.append(measurement_flipped)
	# measurements.append(measurement_flipped)
	# measurements.append(measurement_flipped)
	#measurements.append(measurement_flipped)

	# image_flipped1 = np.fliplr(image1)
	# measurement_flipped1 = -measurement1
	# images.append(image_flipped1)
	# measurements.append(measurement_flipped1)

	# image_flipped2 = np.fliplr(image2)
	# measurement_flipped2 = -measurement2
	# images.append(image_flipped2)
	# measurements.append(measurement_flipped2)

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape = (160, 320, 3)))

model.add(Convolution2D(24,(5,5), padding = 'valid', strides = (2,2), activation='elu'))
model.add(Convolution2D(36,(5,5), padding = 'valid', strides = (2,2), activation='elu'))
model.add(Convolution2D(48,(5,5), padding = 'valid', strides = (2,2), activation='elu'))
model.add(Convolution2D(64,(3,3), padding = 'valid', strides = (1,1), activation='elu'))
model.add(Convolution2D(64,(3,3), padding = 'valid', strides = (1,1), activation='elu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation = 'elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'elu'))
model.add(Dropout(0.5))
model.add(Dense(25))
model.add(Dropout(0.5))
model.add(Dense(1))

#sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='mean_squared_error', optimizer=sgd)
model.compile(loss = 'mse', optimizer = adam)    #mse is used here instead of a cross entropy function because it is a regression and not a classification
model.fit(X_train, y_train, batch_size = 128, validation_split = 0.25, shuffle = True, epochs = 3)

model.save('model.h5')
