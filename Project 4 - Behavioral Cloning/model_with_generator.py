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
from sklearn.model_selection import train_test_split
import sklearn

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


all_samples = []
#data_file = 'orig_data/driving_log.csv'
data_file = 'data/driving_log.csv'
with open(data_file, 'r') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		all_samples.append(line)

data_file = 'data/driving_log_old.csv'
with open(data_file, 'r') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		all_samples.append(line)

random.shuffle(all_samples)
samples = []
i = 0
all_angles = []
for sample in all_samples:
	if (i > 10):
		if (((all_angles[-1]==0)|(all_angles[-2]==0)|(all_angles[-3]==0)|(all_angles[-4]==0)|(all_angles[-5]==0)|(all_angles[-6]==0)|(all_angles[-7]==0)|(all_angles[-8]==0)|(all_angles[-9]==0)|(all_angles[-10]==0))&(float(sample[3])==0)):
			continue
	all_angles.append(float(sample[3]))
	samples.append(sample)

train_samples, validation_samples = train_test_split(samples, test_size=0.25)


def generator(samples, batch_size=128):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		j = 0
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				if(j%500 == 0):
					print(j)
				j+=1
				image = mpimg.imread(batch_sample[0])
				angle = float(batch_sample[3])
				clahe_image = apply_clahe(image)
				brightness_image = modify_brightness(image)
				images.append(image)
				images.append(brightness_image)
				images.append(clahe_image)
				angles.append(angle)
				angles.append(angle)
				angles.append(angle)
				
				image_flipped = np.fliplr(image)
				angle_flipped = -angle
				clahe_image_flipped = apply_clahe(image_flipped)
				brightness_image_flipped = modify_brightness(image_flipped)
				images.append(image_flipped)
				images.append(brightness_image_flipped)
				images.append(clahe_image_flipped)
				angles.append(angle_flipped)
				angles.append(angle_flipped)
				angles.append(angle_flipped)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

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
#model.fit(X_train, y_train, batch_size = 128, validation_split = 0.25, shuffle = True, epochs = 3)
#model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=3)
model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose = 1)
model.save('model.h5')
