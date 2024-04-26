from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.src.utils.module_utils import tensorflow
from sklearn.model_selection import train_test_split
from model.lenet import LeNet
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help='path to input dataset')
ap.add_argument("-m", "--model", required=True, help='path to output model')
args = vars(ap.parse_args())

EPOCHS = 25
INIT_LR = 1e-3
BS = 32

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args['dataset'])))  # paths to input images
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    try:
        image = cv2.imread(imagePath)  # read image
        image = cv2.resize(image, (28, 28))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == 'weapon' else 0
        labels.append(label)
    except:
        continue

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
                                                  random_state=0)  # split into training and test sets

trainY = tensorflow.keras.utils.to_categorical(trainY, num_classes=2)
testY = tensorflow.keras.utils.to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(learning_rate=INIT_LR)  # Removed 'decay' argument
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")

# Removed 'fit_generator', using 'fit' method instead
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // BS,
              epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])  # Changed the extension to '.h5' for model saving
