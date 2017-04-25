#!/usr/bin/python

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from multiprocessing import cpu_count

full_model_weights_path = 'epoch_0.h5'
tuned_full_model_weights_path = 'tuned_model'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'train'
validation_data_dir = 'eval'
nb_train_samples = 10000
nb_validation_samples = 2000
epochs = 10
batch_size = 50

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# convert functional model to sequential, in order to use .add()
model = Sequential(layers=model.layers)

# add the model on top of the convolutional base
model.add(top_model)

# load full model weights
model.load_weights(full_model_weights_path)

for layer in model.layers[:14]:
    layer.trainable = False

model.summary()

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

for i in range(epochs):
    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        max_q_size=100,
        workers=cpu_count(),
        verbose=1,
    )

    # save every 1 epoch
    model.save_weights(tuned_full_model_weights_path + "/epoch_%d.h5" % (i + 1))
