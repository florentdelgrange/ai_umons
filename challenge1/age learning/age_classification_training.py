from time import time

import keras
from keras import backend
from keras.applications import Xception
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, K, Convolution2D, Activation, Flatten, \
    MaxPooling2D
import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.regularizers import l2

def rmse(y_true, y_pred):
    x = backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
    return x

def rmse2(y_true, y_pred):
    return np.abs(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

BATCH_SIZE =20
STEP_PER_EPOCH = 4095//BATCH_SIZE
EPOCH = 10

def generate_dataset(path='men_faces/train', mode='train', rotations=False):

    while 1:
        with open('{}/{}_info.txt'.format(path, mode), 'r') as info:
            batch_step = 0
            # memory optimization
            X = np.empty([BATCH_SIZE, 224, 224, 3])
            LB = np.empty([BATCH_SIZE])
            UB = np.empty([BATCH_SIZE])
            l = []
            for line in info:
                img_name, gender, age = line.split(' ; ')
                l.append(img_name)
                img = load_img('{}/{}'.format(path, img_name), target_size=(224,224))
                x = img_to_array(img)
                X[batch_step] = x
                age_clean = age.strip()
                if("," in age_clean):
                    age = eval(age_clean)
                    LB[batch_step] = age[0]
                    UB[batch_step] = age[1]
                else:
                    age = eval(age_clean)
                    LB[batch_step] = age
                    UB[batch_step] = age

                batch_step += 1

                if batch_step == BATCH_SIZE:
                    yield preprocess_input(X), {'lb_output': LB, 'ub_output': UB}
                    batch_step = 0
                    X = np.empty([BATCH_SIZE, 224, 224, 3])
                    LB = np.empty([BATCH_SIZE])
                    UB = np.empty([BATCH_SIZE])

def getmean(path='men_faces/train',mode='train',):
    X = np.empty([4095, 224, 224, 3])
    batch_step = 0
    with open('{}/{}_info.txt'.format(path, mode), 'r') as info:
        for line in info:
            img_name, gender, age = line.split(' ; ')
            img = load_img('{}/{}'.format(path, img_name), target_size=(224, 224))
            x = img_to_array(img)
            X[batch_step] = x
    print(X.mean(axis=(0, 1, 2)))

def model_training():
    from keras.engine import Model
    from keras.layers import Flatten, Dense, Input
    from keras_vggface.vggface import VGGFace


    
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)

    lb = Dense(32, activation='relu')(x)
    lb = Dense(32, activation='relu')(lb)
    lb = Dense(32, activation='relu')(lb)
    lb_output = Dense(1, activation='relu', name='lb_output')(lb)

    up = Dense(32, activation='relu')(x)
    up = Dense(32, activation='relu')(up)
    up = Dense(32, activation='relu')(up)
    ub_output = Dense(1, activation='relu', name='ub_output')(up)
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=vgg_model.input, outputs=[lb_output, ub_output])
    model.compile(optimizer='adam', loss={'lb_output': 'mse', 'ub_output': 'mse'})
    """
    #history = model.fit_generator(generate_dataset(),steps_per_epoch=STEP_PER_EPOCH,epochs=EPOCH)

    filepath1 = "./weights/y-weights-improvement-{epoch:02d}--{loss:02f}.hdf5"
    checkpoint1 = ModelCheckpoint(filepath1, monitor='loss', verbose=1, save_best_only=True, mode='min')
    filepath2 = "./weights/y-weights-improvement-valoss-{epoch:02d}--{val_loss:02f}.hdf5"
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='{}/logs/{}'.format(".", time()))



    i = Input(shape=(299, 299, 3), name='input')

    x = Conv2D(16, (5,5), strides=(1, 1), padding='valid',
                        activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(i)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x =MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    lb = Dense(150, activation='relu')(x)

    lb_output = Dense(1, activation='relu', name='lb_output')(lb)

    up = Dense(150, activation='relu')(x)
    ub_output = Dense(1, activation='relu', name='ub_output')(up)
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=i, outputs=[lb_output, ub_output])
    model.compile(optimizer='adam', loss={'lb_output': 'mse', 'ub_output': 'mse'})
	"""
    #history = model.fit_generator(generate_dataset(),steps_per_epoch=STEP_PER_EPOCH,epochs=EPOCH)

    filepath1 = "./weights/y-weights-improvement-{epoch:02d}--{loss:02f}.hdf5"
    checkpoint1 = ModelCheckpoint(filepath1, monitor='loss', verbose=1, save_best_only=True, mode='min')
    filepath2 = "./weights/y-weights-improvement-valoss-{epoch:02d}--{val_loss:02f}.hdf5"
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='{}/logs/{}'.format(".", time()))

    
    '''
    i = Input(shape=(299, 299, 3), name='input')
    x = Convolution2D(32, (3, 3), strides = (2, 2), padding = 'valid',
    activation = 'relu', use_bias = True, kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros')(i)
    x = Convolution2D(64, (3, 3), strides=(1,1), padding='valid',
                      activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(
        x)
    x = Flatten()(x)
    x = Dense(100,activation = 'relu')(x)
    x = Dense(100,activation = 'relu')(x)

    output1 = Dense(1, activation='relu', name='output1')(x)
    output2 = Dense(1, activation='relu', name='output2')(x)

    model = Model(inputs=i, outputs=[output1, output2])
    model.compile(optimizer='adam', loss={'output1': 'mse', 'output2': 'mse'})
    '''
    model.fit_generator(generate_dataset(),steps_per_epoch=STEP_PER_EPOCH,epochs=EPOCH,callbacks=[tensorboard, checkpoint1, checkpoint2],validation_data=generate_dataset(path='men_faces/valid', mode="valid"), validation_steps=1350/BATCH_SIZE)
model_training()
'''
import keras.losses
keras.losses.rmse = rmse
model = keras.models.load_model("./weights/weights-improvement-08--253.448055.hdf5")
from matplotlib import  pyplot as plt
import matplotlib.image as mpimg

img = img_to_array(load_img("/home/clement/Desktop/clem.jpg", target_size=(299, 299)))
n = np.empty([1,299,299,3])
n[0] = img
print(model.predict(preprocess_input(n)))
with open("men_faces/valid/valid_info.txt") as f:
    for line in f:
        print(line.split(" ; ")[2])
        img = img_to_array(load_img("men_faces/valid/" + line.split(" ; ")[0], target_size=(299, 299)))
        n = np.empty([1,299,299,3])
        n[0] = img
        print(model.predict(preprocess_input(n)))
        img = mpimg.imread("men_faces/valid/" + line.split(" ; ")[0])
        plt.imshow(img)
        plt.show()
'''
