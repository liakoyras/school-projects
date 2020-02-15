# This file is a duplicate of results_analysis.py in order to
# allow for proper language detection. Please see the original file.
import numpy as np # linear algebra
import pandas as pd # data processing

from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization

# Create the model
# Add convolution and pooling layers
model = models.Sequential()
model.add(layers.Conv2D(64, (11,11), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(128, (11,11), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(256, (11,11), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))
model.add(layers.Dropout(0.2))

# Add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(6, activation='softmax'))

model.summary()

# Cofnigure paths
import os
base_dir = '/kaggle/input/duth-cv-2019-2020-hw-4/vehicles'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen  = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='categorical',
                                                    target_size=(128,128),
                                                    shuffle=True)
# Flow validation images in batches of 20 using val_datagen generator
validation_generator =  val_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='categorical',
                                                         target_size=(128,128))

                                                         from keras.callbacks import ModelCheckpoint

# Define callback that saves the best epoch
mcp_save = ModelCheckpoint('best_epoch.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=5e-5),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      callbacks=[mcp_save],
      verbose=1)


import tensorflow as tf
from keras.preprocessing import image
import csv

model = tf.keras.models.load_model('/kaggle/working/best_epoch.h5')
rowlist = [['Id', 'Category']]

for dirname, _, filenames in os.walk('/kaggle/input/duth-cv-2019-2020-hw-4/vehicles/test'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        img = image.load_img(path, target_size=(128, 128), grayscale=False, interpolation='bilinear')

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        classes_pred = model.predict(x)
        cls_pred = np.argmax(classes_pred)
        rowlist.append([filename, cls_pred])
        with open('output.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rowlist)
