#!/usr/bin/python

from sys import argv

if __name__ == "__main__":
    if len(argv) < 3:
        print("predict the images")
        print("Usage:")
        print("%s <2|3> [filenames] " % argv[0])
        exit(-1)

from time import time
from keras import applications
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import preprocess_input
import math
import numpy as np
import tensorflow as tf


# path to the model weights files.
model_weights_path = 'full_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1))

# convert functional model to sequential, in order to use .add()
model = Sequential(layers=model.layers)

# add the model on top of the convolutional base
model.add(top_model)

model.load_weights(model_weights_path)
#model.summary()


# Save the TF graph to solve the multithread issues
# See also: https://github.com/fchollet/keras/issues/2397
graph = tf.get_default_graph()

def predict(readable):
    global model, graph
    with graph.as_default():

        img = image.load_img(readable, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x *= 1./255
        x = np.expand_dims(x, axis=0)

        try:
            ret = model.predict(x)[0][0]

            cat = 3 if ret > 0 else 2

        except Exception, e:
            print e.message

        return cat, ret

if __name__ == "__main__":
    cat = int(argv[1])
    fn = argv[2:]
    hit = 0

    for i in fn:
        pred_cat, ret = predict(i)

        if pred_cat == cat:
            hit += 1

        print "%20s: %5s (S = %.4f)" % (
            i,
            "Right" if cat == pred_cat else "Wrong",
            ret,
        )

    print "[%d/%d] hit (%.3f%%)" % (hit, len(fn), 100. * hit / len(fn))

