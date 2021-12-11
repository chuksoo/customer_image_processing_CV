
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def load_train(path):
    """
    It loads the train part of dataset from path
    """
    # data generator
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=90)

    # extract data from the directory
    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345)

    return train_gen_flow


def load_test(path):
    """
    It loads the validation/test part of dataset from path
    """
    # validation data generator
    test_datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255)

    # extract data from the directory
    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345)

    return test_gen_flow


def create_model(input_shape):
    """
    It defines the model
    """
    backbone = ResNet50(
        input_shape=(150, 150, 3), weights='imagenet', include_top=False
    )

    # define the model
    model = Sequential()
    
    # add layers to model
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(150, activation='relu'))
    model.add(Dense(85, activation='relu'))
    model.add(Dense(12, activation='relu'))
    
    # add compiler
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error', metrics=['mean_absolute_error']
                  )

    return model


def train_model(
    model, 
    train_data, 
    test_data, 
    batch_size=None, 
    epochs=20,
    steps_per_epoch=None, 
    validation_steps=None
):
    """
    Trains the model given the parameters
    """
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=10,
        verbose=2,
        #steps_per_epoch=7,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    return model


