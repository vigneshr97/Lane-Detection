import pickle
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
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


 ### Train your model here.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
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
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
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
new_image_files = glob.glob('traffic-signs-data/new*.jpg')
new_img = np.zeros((10,32,32,3))
for idx, fname in enumerate(new_image_files):
    new_img[i] = cv2.imread(fname)
    print(LeNet(new_img[i]))









def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
