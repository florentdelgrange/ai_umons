from keras.layers import Dense, Conv2D, MaxPooling2D, Concatenate, Flatten, GlobalAveragePooling2D, Dropout, Activation, Input, Lambda
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from time import time
from keras import backend
import numpy as np
from keras.models import Model, Sequential
from model_training import save
from scipy.ndimage.interpolation import rotate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from docopt import docopt
import os
import PIL.Image


input_shape = (299, 299, 3)

EPOCHS = 42
BATCH_SIZE = 21
STEPS_PER_EPOCH = 9146 // BATCH_SIZE
CUSTOM_SAVE_PATH = '.'

def preprocess_input(x):
    x = np.array(x, dtype='float')
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def generate_dataset(path='sorted_faces/train', mode='train'):
    datagen_openu = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.3,
            fill_mode='nearest'
            )
    while 1:
        with open('{}/{}_info.txt'.format(path, mode), 'r') as info:
            batch_step = 0
            X = np.empty([BATCH_SIZE, 299, 299, 3])
            Y = np.empty([BATCH_SIZE])
            for line in info:
                angle = np.random.uniform(-90, 90)
                #angle = np.random.randint(-90, 90)
                img_name, _, age = line.split(' ; ')

                age = eval(age)
                # If age is not well labelised (None in dataset), it takes the value 255
                if not age:
                    age = 255
                if type(age) == tuple:
                    age = np.mean(age)
                if age == 255:
                    pass
                else:
                    img = load_img('{}/all/{}'.format(path, img_name), target_size=(299, 299))#.rotate(angle, resample='1')
                    #img.show()
                    x = img_to_array(img)
                    X[batch_step] = x
                    Y[batch_step] = age
                    batch_step += 1

                    if batch_step == BATCH_SIZE:
                        X, Y = datagen_openu.flow(x=X, y=Y, batch_size=BATCH_SIZE).next()
                        #PIL.Image.fromarray(np.array(X[0], dtype='uint8')).show()
                        #print(Y[0])
                        yield (preprocess_input(X), Y)
                        batch_step = 0
                        X = np.empty([BATCH_SIZE, 299, 299, 3])
                        Y = np.empty([BATCH_SIZE])

if __name__=='__main__':
    json_file = open('{}/models/xception_gender.json'.format(CUSTOM_SAVE_PATH), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    gender_model = model_from_json(loaded_model_json)
    # load weights into new model
    gender_model.load_weights("{}/models/xception_gender.h5".format(CUSTOM_SAVE_PATH))

    np.random.seed(42)

    input = Input(shape=(299, 299, 3))
    x = Conv2D(16, kernel_size=(7, 7), strides=(1, 1),
                     activation='relu')(input)
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    subnets = [None] * 2
    for i in range(2):
        y = Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu')(x)
        y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu')(x)
        y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)
        y = Flatten()(y)
        print(gender_model(input)[0][0])
        if i == 0:
            y = Lambda(lambda x : (1 - gender_model(input)[0][0]) * x)(y)
        if i == 1:
            y = Lambda(lambda x : gender_model(input)[0][0] * x)(y)
        y = Dense(299, activation='relu')(y)
        subnets[i] = Dropout(0.15)(y)
    x = Concatenate(subnets)
    x = Dense(100, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    prediction = Lambda(int)(x)

    model = Model(inputs=input, outputs=prediction)

    # Fit
    model.fit_generator(generate_dataset(), steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=generate_dataset(path='sorted_faces/valid', mode='valid'),
                        validation_steps=30,
                        epochs=EPOCHS)

    save(model, path=CUSTOM_SAVE_PATH, model_name='age_regressor')
