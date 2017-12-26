"""Train a rotation regressor

Usage:
    rotation_regressor_training.py [--load_model <model_name>] [--add_dropout]
    rotation_regressor_training.py (-h | --help)

Options:
-h --help                                    Display help.
--load_model <model_name>                    Train a model already saved.
--add_dropout                                Add Dropout Layer after the first pooling.

"""
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout, Activation
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

def generate_dataset(path='sorted_faces/train', mode='train'):
    while 1:
        with open('{}/{}_info.txt'.format(path, mode), 'r') as info:
            batch_step = 0
            X = np.empty([BATCH_SIZE, 299, 299, 3])
            Y = np.empty([BATCH_SIZE])
            for line in info:
                angle = np.random.uniform(-90, 90)
                #angle = np.random.randint(-90, 90)
                img_name, gender, age = line.split(' ; ')
                img = load_img('{}/all/{}'.format(path, img_name), target_size=(299, 299))#.rotate(angle, resample='1')
                #img.show()
                x = img_to_array(img)
                x = np.array(x, dtype='uint8')
                x = rotate(x, angle, mode='nearest', reshape=False)
                X[batch_step] = x
                Y[batch_step] = angle
                batch_step += 1

                if batch_step == BATCH_SIZE:
                    #PIL.Image.fromarray(np.array(X[0], dtype='uint8')).show()
                    #print(Y[0])
                    yield (preprocess_input(X), Y)
                    batch_step = 0
                    X = np.empty([BATCH_SIZE, 299, 299, 3])
                    Y = np.empty([BATCH_SIZE])

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

if __name__=='__main__':
    args = docopt(__doc__)
    if not args['--load_model']:
        np.random.seed(42)
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(150, activation='relu'))
        model.add(Dense(1))#, kernel_initializer='normal'))
    else:
        json_file = open('models/{}.json'.format(args['--load_model']), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("models/{}.h5".format(args['--load_model']))
        if args['--add_dropout']:
            pred = model.output
            x = model.layers[1].output
            x = Dropout(0.15)(x)
            for layer in model.layers[2:]:
                x = layer(x)
            # Create a new model
            model2 = Model(input=model.input, output=x)
            model = model2
        model.summary()


    model.compile(loss='mse', optimizer='nadam', metrics=[rmse, 'mean_squared_error'])
    if not os.path.exists('{}/weights'.format(CUSTOM_SAVE_PATH)):
        os.makedirs("{}/weights".format(CUSTOM_SAVE_PATH))
    
    filepath= CUSTOM_SAVE_PATH + "/weights/rotationreg-weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='{}/logs/rotation-reg-{}'.format(CUSTOM_SAVE_PATH, time()))#, histogram_freq=1, write_grads=True, batch_size=BATCH_SIZE)

    # Fit
    model.fit_generator(generate_dataset(), steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=generate_dataset(path='sorted_faces/valid', mode='valid'),
                        validation_steps=30,
                        epochs=EPOCHS, callbacks=[tensorboard, checkpoint])

    save(model, path=CUSTOM_SAVE_PATH, model_name='rotation_regressor_2')

