# cnn_model_train.py (updated)
import os
import warnings
from absl import logging as absl_logging
import numpy as np
import pickle
import cv2
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', message='.*tf.reset_default_graph.*')
absl_logging.set_verbosity(absl_logging.ERROR)

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

# SETTINGS (keep in sync with final.py)
IMAGE_SIZE = (50, 50)  # (height, width)
NORMALIZE_INPUT = True

def get_image_size():
    return IMAGE_SIZE

image_x, image_y = get_image_size()

def get_num_of_classes():
    return len(glob('gestures/*'))

def cnn_model(num_of_classes: int):
    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape=(image_x, image_y, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "cnn_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return model, [checkpoint1]

def train():
    # read pickled arrays
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    # reshape to (n,h,w,1) and normalize if configured
    train_images = train_images.reshape((train_images.shape[0], image_x, image_y, 1)).astype(np.float32)
    val_images = val_images.reshape((val_images.shape[0], image_x, image_y, 1)).astype(np.float32)
    if NORMALIZE_INPUT:
        train_images /= 255.0
        val_images /= 255.0

    num_classes = int(max(train_labels.max(), val_labels.max())) + 1
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)

    model, callbacks_list = cnn_model(num_classes)
    model.summary()
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=20, batch_size=64, callbacks=callbacks_list)
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

if __name__ == "__main__":
    train()
    tf.keras.backend.clear_session()
