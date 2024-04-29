import numpy as np
import argparse
import os
import cv2
import random

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from imutils import paths


gpus = tf.config.experimental.list_physical_devices('GPU') #get gpus
if gpus: #if you have a gpu
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) #use gpu optimization
    except RuntimeError as e:
        print("RuntimeError: ", e)

def build_model(width, height, depth, classes): 
    inputShape = (height, width, depth) #set shape of input data which defines the input layer of the network
    inputs = Input(shape=inputShape) #creates the input layer of the neural net
    x = Conv2D(20, (5, 5), padding="same", activation="relu")(inputs) #applies a 2d convolutional neural network with 20 filters
    x = MaxPooling2D(pool_size=(2, 2))(x) #adds a 2D max pooling layer with a 2x2 window to reduce the spatial dimension of the output
    x = Conv2D(50, (5, 5), padding="same", activation="relu")(x) #applies a 2d convolutional neural network with 50 filters
    x = MaxPooling2D(pool_size=(2, 2))(x)  #adds a 2D max pooling layer with a 2x2 window to reduce the spatial dimension of the output
    x = Flatten()(x) #flattens the multi dimensional input into a  ingle dimension.
    x = Dense(500, activation="relu")(x) #applies a 2d convolutional neural network with 500 filters
    outputs = Dense(classes, activation="softmax")(x) #defines the output layer
    
    model = Model(inputs=inputs, outputs=outputs) #constructs the model using the input and output layer.
    return model #returns the model

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help='path to input dataset') #get the path of the datasets
ap.add_argument("-m", "--model", required=True, help='path to output model') #get path of the output models
args = vars(ap.parse_args()) #get the args

EPOCHS = 10 #number of epochs ie iterations
INIT_LR = 1e-3 #adam optimizer learning rate
BS = 32

print("[INFO] loading images...")
data = [] #this is the data list
labels = [] #this is a list of the labels
classLabels = [] #this is a list of class labels we will be using.

imagePaths = sorted(list(paths.list_images(args['dataset']))) #get the paths of the folders in the data folder
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths: #gets the path of each image
    try:
        image = cv2.imread(imagePath) #read the image path
        image = cv2.resize(image, (256, 256)) #resize images to 356x356
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2] #this is to get the labels of the images
        if label not in classLabels: #if it doesnt already have that label
            classLabels.append(label) # add a new label
        labels.append(classLabels.index(label)) #append the labels to the class labels
    except:
        continue

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
num_classes = len(classLabels) #sets the number of classes to the amount of classes in num_classes

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=0) #sets train and test split so that 25% of image will be used for testing and 75% will be for training

trainY = to_categorical(trainY, num_classes=num_classes) #turns class labels into encoded format
testY = to_categorical(testY, num_classes=num_classes)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = build_model(256, 256, 3, num_classes) #build the model with the image dimensions, 
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) #compile the model

print("[INFO] training network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // BS,
              epochs=EPOCHS, verbose=1) #trains the model using the augmented data

print("[INFO] serializing network...")
model.save(args["model"]) #saves the model to the file specified