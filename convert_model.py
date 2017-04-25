#!/usr/bin/python

from sys import argv

if __name__ == "__main__":
    if len(argv) != 3:
        print("convert top model to full model")
        print("Usage:")
        print("%s top_model.h5 full_model.h5" % argv[0])
        exit(-1)

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

if __name__ == "__main__":
    top_model_weights_path = argv[1]
    full_model_weights_path = argv[2]
    # dimensions of our images.
    img_width, img_height = 224, 224

    # build the VGG16 network
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.load_weights(top_model_weights_path)

    # convert functional model to sequential, in order to use .add()
    model = Sequential(layers=model.layers)

    # add the model on top of the convolutional base
    model.add(top_model)

    model.save_weights(full_model_weights_path)

    model.summary()
