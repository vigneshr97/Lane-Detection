# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, an autonomous vehicle is trained to drive on its own with a hard coded speed value which is 10 mph and a steering angle obtained from the model developed. The model, 'model.h5' was developed by training an autonomous vehicle by steering it in a simulator and running the data through a deep neural network using Keras with an architecture similar to NVIDIA DNN architecture. The architecture was adopted with minor tweaks to suit the problem, a technique known as Transfer Learning.

The project has five files: 
* model.py  - script used to create and train the model
* drive.py  - script to drive the car
* model.h5  - a trained Keras model obtained using model.py
* model_with_generator.py - Has almost the same thing as model.py except that it uses a generator function
* writeup_report.md - a report writeup file
* output.mp4 - a video recording of the vehicle driving autonomously around the track for one lap

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 output
```

The fourth argument, `output`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py output
```

Creates a video based on images found in the `output` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `output.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 60
```

Will run the video at 60 FPS. The default FPS is 60.


Explaining the Code
---

The images are shuffled in the very beginning to remove too continuous straight driving. Since the angle is biased a lot towards 0, it is ensured that there doesn't exist more than 1 zero in every 10 continuous image except for the first 10 by removing the straight driving images. The data is augmented by using histogram equalization and brightness modification. The images are flipped and the same augmentation techniques were used. The image is cropped 50 px in the top and 20 px in the bottom due to redundant data. The images are normalized to lie between -0.5 and 0.5. The architecture is similar to the architecture that NVIDIA used for its self driving cars as the task is similar. It starts with convolutional layers that have 5x5 filters with depths of 24, 36 and 48 respectively in the first 3 layers and a stride of 2 in both the directions. These layers are followed by two convolutional layers each having 3x3 filters of depth 64 with strides of 1 in both the directions. Exponential Linear Unit activation function is used in all the five convolutional layers. A maxpooling layer follows. The layer is flattened and fed into four fully connected layers with first two having ELU activation. Dropout is applied in each layer after the first one with a probability of retaining the weights equal to 0.5. It was also realized that maxpooling layers take a long time. SGD was tried as the optimizer. But Adam optimizer performed better. An Adam optimizer with a modifiable learning rate was used. The learning rate was set to 0.001. Loss function was mean squared error as the objective is regression. A model named model.h5 was created after training around 42000 images and validating against around 14000 images. The bath size was 128 and the number of epochs was 3. Since generator function consumed a lot of time, the model available in the directory was created without a generator function. model_with_generator.py has the code with a generator function
