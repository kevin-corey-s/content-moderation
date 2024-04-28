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


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("RuntimeError: ", e)

def build_model(width, height, depth, classes):
    inputShape = (height, width, depth)
    inputs = Input(shape=inputShape)
    x = Conv2D(20, (5, 5), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(50, (5, 5), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(500, activation="relu")(x)
    outputs = Dense(classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help='path to input dataset')
ap.add_argument("-m", "--model", required=True, help='path to output model')
args = vars(ap.parse_args())

EPOCHS = 10
INIT_LR = 1e-3
BS = 32

print("[INFO] loading images...")
data = []
labels = []
classLabels = []

imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    try:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        if label not in classLabels:
            classLabels.append(label)
        labels.append(classLabels.index(label))
    except:
        continue

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
num_classes = len(classLabels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=0)

trainY = to_categorical(trainY, num_classes=num_classes)
testY = to_categorical(testY, num_classes=num_classes)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = build_model(256, 256, 3, num_classes)
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // BS,
              epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])