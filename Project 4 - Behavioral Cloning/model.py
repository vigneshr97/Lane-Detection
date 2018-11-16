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
#The images are shuffled in the very beginning to remove too continuous straight driving
images = []
measurements = []
i = 0
for line in lines:
	#since the angle is biased a lot towards 0, it is ensured that there doesn't exist more than 1 zero in every 10 continuous image except for the first 10.
	if i > 10:
		if (((measurements[-1]==0)|(measurements[-2]==0)|(measurements[-3]==0)|(measurements[-4]==0)|(measurements[-5]==0)|(measurements[-6]==0)|(measurements[-7]==0)|(measurements[-8]==0)|(measurements[-9]==0)|(measurements[-10]==0))&(float(line[3])==0)):
			continue

	# if i%500 == 0:
	# 	print(i)
	i += 1
	#The image is read using mpimg to obtain it in RGB format
	image = mpimg.imread(line[0])
	measurement = float(line[3])
	images.append(image)
	measurements.append(measurement)
	
	#The data is augmented by using histogram equalization and brightness modification
	clahe_image = apply_clahe(image)
	brightness_image = modify_brightness(image)
	images.append(brightness_image)
	images.append(clahe_image)
	measurements.append(measurement)
	measurements.append(measurement)

	#The image is flipped and the same augmentations as above are performed
	image_flipped = np.fliplr(image)
	measurement_flipped = -measurement
	images.append(image_flipped)
	measurements.append(measurement_flipped)

	clahe_image_flipped = apply_clahe(image_flipped)
	brightness_image_flipped = modify_brightness(image_flipped)
	images.append(clahe_image_flipped)
	images.append(brightness_image_flipped)
	measurements.append(measurement_flipped)
	measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
#The image is cropped 50 px in the top and 20 px in the bottom due to redundant data
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
#The images are normalized to lie between -0.5 and 0.5.
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape = (160, 320, 3)))

#The architecture is similar to NVIDIA architecture. It starts with convolutional layers that have 
#24, 36 and 48 , 5x5 filters respectively in the first 3 layers and a stride of 2 in both the directions.
#These layers are followed by two convolutional layers each having 64 3x3 filters with strides of 1
#in both the directions. Exponential Linear Unit activation function is used in all the five convolutional
#layers. A maxpooling layer follows. The layer is flattened and fed into four fully connected layers with
#first two having ELU activation. Dropout is applied in each layer after the first one with a probability of
#retaining the weights equal to 0.5. It was also realized that maxpooling layers take a long time
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
#SGD was tried. But Adam optimizer performed better. An Adam optimizer with a modifiable learning rate
#was used. The learning rate was set to 0.001. Loss function was mean squared error as the objective
#is regression. A model named model.h5 was created after training around 20000 images and validating
#against around 6000 images. The bath size was 128 and the number of epochs was 3.
#Since generator function consumed a lot of time, the model available in the directory was created without
#a generator function. model_with_generator.py has the code with a generator function
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='mean_squared_error', optimizer=sgd)
model.compile(loss = 'mse', optimizer = adam)    #mse is used here instead of a cross entropy function because it is a regression and not a classification
model.fit(X_train, y_train, batch_size = 128, validation_split = 0.25, shuffle = True, epochs = 3)

model.save('model.h5')
