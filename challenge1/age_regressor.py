from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from time import time
from keras import backend
import numpy as np
from keras.applications.xception import Xception
from keras.models import Model, Sequential
from model_training import save
from scipy.ndimage.interpolation import rotate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from docopt import docopt
import tensorflow as tf
import os
import PIL.Image


input_shape = (299, 299, 3)
DATABASE_SIZE = 43390
EPOCHS = 1
BATCH_SIZE = 20
STEPS_PER_EPOCH = DATABASE_SIZE // (10 * BATCH_SIZE)
VALIDATION_STEPS = STEPS_PER_EPOCH // 5
gender_dict = {'m': 0, 'f' : 1}
CUSTOM_SAVE_PATH = '/home/florent/Dropbox/Info/ai_umons/challenge1'

def preprocess_input(x):
    x = np.array(x, dtype='float32')
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def generate_dataset(path='sorted_faces/train', mode='train'):
    datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.3,
            fill_mode='nearest'
            )
    while 1:
        if mode == 'train':
            length = 43390
            X_train = np.memmap('{}/training_images'.format('Dataset'), dtype='uint8', mode='r', shape=(length, 299, 299, 3))
            Y_train = np.memmap('{}/training_ages'.format('Dataset'), dtype='uint8', mode='r', shape=(length))
            for i in range(length // BATCH_SIZE):
                X = np.array(X_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], dtype='uint8')
                Y = np.array(Y_train[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], dtype='uint8')
                X, Y = datagen.flow(x=X, y=Y, batch_size=BATCH_SIZE,
                        ).next()
                yield preprocess_input(X), Y
        elif mode == 'valid':
            #length = 6943
            length = VALIDATION_STEPS * BATCH_SIZE
            X_test = np.memmap('{}/testing_images'.format('Dataset'), dtype='uint8', mode='r', shape=(length, 299, 299, 3))
            Y_test = np.memmap('{}/testing_ages'.format('Dataset'), dtype='uint8', mode='r', shape=(length))
            for i in range(length // BATCH_SIZE):
                X = np.array(X_test[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], dtype='uint8')
                Y = np.array(Y_test[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], dtype='uint8')
                yield preprocess_input(X), Y

if __name__=='__main__':
    json_file = open('{}/models/fine_tuned_xception_gender.json'.format(CUSTOM_SAVE_PATH), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    gender_model = model_from_json(loaded_model_json)
    # load weights into new model
    gender_model.load_weights("{}/models/fine_tuned_xception_gender.h5".format(CUSTOM_SAVE_PATH))
    for layer in gender_model.layers:
        layer.trainable = False

    base_model = Xception(include_top=False, input_shape=(299, 299, 3))
    input = base_model.input
    for layer in base_model.layers[:85]:
       layer.trainable = False

    x = base_model.layers[95].output

    subnets = [None] * 2
    for i, gender in enumerate(['male', 'female']):
        prefix=gender
        y = x
        for j in range(2):
            residual = y
            y = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_1sepconv{}'.format(j))(y)
            y = BatchNormalization(name=prefix + '_sepconv1_{}_bn'.format(j))(y)
            y = Activation('relu', name=prefix + '_sepconv2_{}_act'.format(j))(y)
            y = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_2sepconv{}'.format(j))(y)
            y = BatchNormalization(name=prefix + '_sepconv32_{}_bn'.format(j))(y)
            y = Activation('relu', name=prefix + '_sepconv4_{}_act'.format(j))(y)
            y = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_3sepconv{}'.format(j))(y)
            y = BatchNormalization(name=prefix + '_sepconv3_{}_bn'.format(j))(y)

            y = Add()([y, residual])

        y = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='{}block{}_pool'.format(gender, j))(y)
        y = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name=prefix+'block14_sepconv1')(y)
        y = BatchNormalization(name=prefix + 'block14_sepconv1_bn')(y)
        y = Activation('relu', name=prefix+'block14_sepconv1_act')(y)
        y = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name=prefix+'block14_sepconv2')(y)
        y = BatchNormalization(name=prefix+'block14_sepconv2_bn')(y)
        y = Activation('relu', name=prefix+'block14_sepconv2_act')(y)

        y = GlobalAveragePooling2D(name=prefix+'_global_average_pooling2D')(y)
        if i == 0:
            # Multiply by men probability
            z = gender_model(input)
            z = Lambda(lambda x: 1 - x)(z)
            y = Multiply(name='{}_proba_multiply'.format(gender))([z, y])
        if i == 1:
            # Multiply by female probability
            z = gender_model(input)
            y = Multiply(name='{}_proba_multiply'.format(gender))([z, y])
        y = Dense(299, activation='relu')(y)
        subnets[i] = Dropout(0.25)(y)
        subnets[i] = y
    x = Concatenate()(subnets)
    x = Dense(100, activation='relu')(x)
    prediction = Dense(1)(x)
    #prediction = Lambda(lambda i: tf.floor(i))(x)
    #prediction = Activation(int)(x)

    model = Model(inputs=input, outputs=prediction)

    model.compile(loss='mse', optimizer='nadam', metrics=[rmse, 'mean_squared_error'])
    model.summary()

    from keras.utils import plot_model
    plot_model(model, to_file='age_regressor.png')

    # Fit
    model.fit_generator(generate_dataset(), steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=generate_dataset(path='sorted_faces/valid', mode='valid'),
                        validation_steps=VALIDATION_STEPS,
                        epochs=EPOCHS)

    save(model, path=CUSTOM_SAVE_PATH, model_name='age_regressor_1')
