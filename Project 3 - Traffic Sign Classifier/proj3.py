import pickle
import numpy as np
import random
import cv2
import numpy as np
import matplotlib as mpl
import pandas as pd
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from scipy import misc
from scipy.ndimage import rotate
from skimage import transform
from skimage.transform import warp, SimilarityTransform, AffineTransform
from skimage import exposure
from sklearn.utils import shuffle
%matplotlib inline

# Load pickled data
training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
label_train, count_train = np.unique(np.array(y_train), return_counts=True)
label_valid, count_valid = np.unique(np.array(y_valid), return_counts=True)
label_test, count_test = np.unique(np.array(y_test), return_counts=True)
plt.rcParams.update({'font.size': 22}) 
# create plot
fig, ax = plt.subplots(figsize = (20,15))
index = np.arange(n_classes)
bar_width = 0.5
opacity = 0.8
 
rects1 = plt.bar(index, count_train, bar_width, alpha = opacity, color = 'b', label = 'train')

plt.xlabel('Category')
plt.ylabel('Number of datapoints')
plt.title('Training set visualization')
plt.xticks(index, label_train)
plt.legend()
 
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (20,15))
index = np.arange(n_classes)
bar_width = 0.5
opacity = 0.8
 
rects2 = plt.bar(index, count_valid, bar_width, alpha = opacity, color='g', label='valid')
 
plt.xlabel('Category')
plt.ylabel('Number of datapoints')
plt.title('Validation set visualization')
plt.xticks(index, label_train)
plt.legend()
 
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (20,15))
index = np.arange(n_classes)
bar_width = 0.5
opacity = 0.8
 
rects3 = plt.bar(index, count_test, bar_width, alpha = opacity, color='r', label='test')
 
plt.xlabel('Category')
plt.ylabel('Number of datapoints')
plt.title('Test set visualization')
plt.xticks(index, label_train)
plt.legend()
 
plt.tight_layout()
plt.show()

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE()
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    c = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return c.astype(np.uint8)

def apply_rotation(img):
    rotated = 255*transform.rotate(img, angle=np.random.uniform(-15, 15), mode='edge')
    rotated = rotated.astype(np.uint8)
    return rotated.astype(np.uint8)

def apply_translation(img):
    translated = 255*warp(img, transform.SimilarityTransform(translation=(np.random.uniform(-5, 5), np.random.uniform(-5, 5))),mode='edge')
    translated = translated.astype(np.uint8)
    return translated.astype(np.uint8)

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
    
def augment1(img, k):
    if(k == 1):
        return apply_rotation(apply_translation(img))
    if(k == 2):
        return apply_clahe(apply_rotation(img))
    if(k == 3):
        return modify_brightness(apply_translation(img))
    if(k == 4):
        return modify_brightness(apply_rotation(img))
    if(k == 5):
        return modify_brightness(apply_clahe(img))
    if(k == 6):
        return apply_clahe(apply_translation(img))
    if(k == 7):
        return apply_rotation(img)
    if(k == 8):
        return apply_translation(img)

def augment2(img, k):
    if(k == 1):
        return medianblur(apply_translation(img))
    if(k == 2):
        return gaussianblur(apply_translation(img))
    if(k == 3):
        return medianblur(apply_rotation(img))
    if(k == 4):
        return gaussianblur(apply_rotation(img))

print('Data augmentation on progress .......')
augment_X = []
augment_y = []

labels, count = np.unique(np.array(y_train), return_counts = True)
for i in range(len(labels)):
    label = labels[i]
    c = count[i]
    if c > 2500:
        continue
    while c < 2500:
        indices = np.where(y_train==label)[0]
        index1 = np.random.choice(indices)
        index2 = np.random.choice(indices)
        rand1 = np.random.randint(1,9)
        rand2 = np.random.randint(1,5)
        augment_X.append(augment1(X_train[index1], rand1))
        augment_X.append(augment2(X_train[index2], rand2))
        augment_y.append(y_train[index1])
        augment_y.append(y_train[index2])
        c += 2

X_train = np.append(X_train, augment_X).reshape(len(X_train)+len(augment_X),32,32,3)
y_train = np.append(y_train, augment_y)

X_train = np.dot(X_train[:,:,:,0:3],[0.299, 0.587, 0.114]).reshape(len(X_train),32,32,1)
X_valid = np.dot(X_valid[:,:,:,0:3],[0.299, 0.587, 0.114]).reshape(4410,32,32,1)
X_test = np.dot(X_test[:,:,:,0:3],[0.299, 0.587, 0.114]).reshape(12630,32,32,1)

X_train = X_train/128 - 1
X_valid = X_valid/128 - 1
X_test = X_test/128 - 1

n_train = len(X_train)
print("Number of training examples after augmentation=", n_train)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 100
BATCH_SIZE = 256

### Define your architecture here.
### Feel free to use as many code cells as needed.
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.06
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x16.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(16))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation
    conv1 = tf.nn.relu(conv1)

    # Pooling: Input = 28x28x16. Output = 14x14x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation
    conv2 = tf.nn.relu(conv2)

    # Pooling: Input = 10x10x64. Output = 5x5x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten: Input = 5x5x64. Output = 1600.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 1600. Output = 400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1600, 400), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation and a dropout with a random keep_prob
    fc1    = tf.nn.relu(fc1)
    fc1    = tf.nn.dropout(fc1,random.random())

    # Layer 4: Fully Connected. Input = 400. Output = 200.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(400, 200), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(200))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation and a dropout with a random keep_prob
    fc2    = tf.nn.relu(fc2)
    fc2    = tf.nn.dropout(fc2,random.random())

    # Layer 5: Fully Connected. Input = 200. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(200, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

### Calculate and report the accuracy on the training and validation set.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def predict(X_data):
    sess = tf.get_default_session()
    prediction = sess.run(logits, feed_dict = {x: X_data})
    return prediction

train_accuracy = []
valid_accuracy = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        validation_accuracy = evaluate(X_valid, y_valid)
        training_accuracy = evaluate(X_train, y_train)
        valid_accuracy.append(validation_accuracy)
        train_accuracy.append(training_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    df=pd.DataFrame({'epoch': range(1,101), 'Training_accuracy': train_accuracy, 'Validation_accuracy': valid_accuracy})
    plt.plot( 'epoch', 'Training_accuracy', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    plt.plot( 'epoch', 'Validation_accuracy', data=df, marker='', color='olive', linewidth=2)
    plt.legend()
    saver.save(sess, './lenet')
    print("Model saved")

### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
### Feel free to use as many code cells as needed.

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
#from PIL import Image
import csv
new_image_files = ['60kmph.jpg', 'bicycles crossing.jpg', 'road work.jpg', 'slippery road.jpg', 'stop sign.jpg']
new_data = np.asarray([3, 29, 25, 23, 14])
new_img = []
i = 0
sign_name = []
with open('signnames.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for sign in reader:
        sign_name.append(sign[1])
    
for fname in new_image_files:
    img = mpimg.imread('traffic-signs-data/'+fname)
    plt.figure()
    plt.imshow(img)
    img = cv2.resize(img, (32, 32))
    img = np.dot(img[:,:,0:3],[0.299, 0.587, 0.114]).reshape(img.shape[0],img.shape[1],1)
    img = img/128 - 1
    new_img.append(np.float32(img))
new_img = np.asarray(new_img)

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
accuracy = 0
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    prediction_logits = predict(new_img)
    exp_pred = np.exp(prediction_logits)
    s_pred = np.sum(exp_pred, axis = 1).reshape(5, 1)
    prediction = exp_pred/s_pred
    top5 = sess.run(tf.nn.top_k(tf.constant(prediction), k=5))
    print(top5)
    print()
    print()
    print('Predictions:')
    predictions = top5.indices[:,0]
    accuracy = np.sum(predictions == new_data)
    sign_predictions = []
    for i in range(len(predictions)):
        sign_predictions.append(sign_name[predictions[i]])
    print(sign_predictions)
#     accuracy = evaluate(new_img, new_data)
    accuracy = accuracy*100/5
    print("Test Accuracy = {:.3f}".format(accuracy))
    
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
print("New images Accuracy = {:.2f}%".format(accuracy))

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
pred = top5.indices
pred_sign_names = []
for i in range(len(pred)):
    pred_sign_names.append([])
    for j in range(len(pred[i])):
        pred_sign_names[i].append(sign_name[pred[i][j]])

for i in range(5):
    print('probabilities on the image '+new_image_files[i][:-4]+':')
    for j in range(5):
        print('probability of the image being '+str(pred_sign_names[i][j])+': {:.3f}'.format(top5.values[i][j]))
    print('')
    print('')
print(top5.values)
