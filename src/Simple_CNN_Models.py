import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D,ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split


def Simple_CNN_Classifier(train_datagen_gender, validation_datagen_gender):
    model = Sequential()

    model.add(Convolution2D(filters = 64, kernel_size = (3, 3), padding = 'same'
                            ,input_shape=(100,100,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(rate = 0.5))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(rate = 0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    checkpoint = ModelCheckpoint('simple_cnn_gender.h5', monitor = 'val_loss', 
                             mode = 'min', save_best_only = True, verbose = 1)
    earlystopping = EarlyStopping(monitor= 'val_loss', min_delta= 0, patience = 10, 
                              verbose = 1, restore_best_weights= True)   
    history_gender = model.fit(train_datagen_gender, batch_size=64, epochs=50, verbose=2,
                    validation_data=validation_datagen_gender,callbacks=[earlystopping, checkpoint])

    return history_gender


def Simple_CNN_Regressor(train_datagen_age, validation_datagen_age):
    model = Sequential()
    model.add(Convolution2D(filters = 64, kernel_size = (3, 3), padding = 'same',
                            input_shape=(100,100,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Convolution2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'linear'))

    opt = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mse'])

    print(model.summary())

    checkpoint = ModelCheckpoint('simple_cnn_age.h5', monitor = 'val_loss', 
                             mode = 'min', save_best_only = True, verbose = 1)
    earlystopping = EarlyStopping(monitor= 'val_loss', min_delta= 0, patience = 10, 
                              verbose = 1, restore_best_weights= True)
    history_age = model.fit(train_datagen_age, batch_size=64, epochs=50, verbose=2,
                    validation_data=validation_datagen_age,callbacks=[earlystopping, checkpoint])
    
    return history_age


if __name__ == '__main__':
    imggen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1/255.0,validation_split=0.2)
    batch_size=32
    target_size = (100,100)

    # Gender Detection
    filepath_gender = '/home/ubuntu/data/AFAD-Full/AFAD_gender'

    train_datagen_gender = imggen.flow_from_directory(filepath_gender, target_size=target_size,
                                           class_mode = 'binary',batch_size=batch_size,subset='training')
    validation_datagen_gender = imggen.flow_from_directory(filepath_gender, target_size=target_size,
                                            class_mode = 'binary',batch_size=batch_size, subset='validation')

    history_gender = Simple_CNN_Classifier(train_datagen_gender, validation_datagen_gender)

    # Age Estimation
    filepath_age = '/home/ubuntu/data/AFAD-Full/AFAD'

    train_datagen_age = imggen.flow_from_directory(filepath_age, target_size=target_size,batch_size=batch_size, 
                                           class_mode = 'sparse',subset='training')
    validation_datagen_age = imggen.flow_from_directory(filepath_age, target_size=target_size, batch_size=batch_size,
                                           class_mode = 'sparse', subset='validation')

    history_age = Simple_CNN_Regressor(train_datagen_age, validation_datagen_age)


    