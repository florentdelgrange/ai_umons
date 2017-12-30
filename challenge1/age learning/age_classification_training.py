from time import time

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from keras_vggface.vggface import VGGFace
from sklearn.preprocessing import LabelEncoder

NBR_OF_CLASSES = 8
TRAINING_SET_SIZE = 4096
VALIDATION_SET_SIZE = 1366
IMAGE_SIZE = 224
BATCH_SIZE = 48
STEP_PER_EPOCH = 4096 // BATCH_SIZE
EPOCH = 10

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(['(15, 20)', '(4, 6)', '(48, 53)', '(25, 32)', '(0, 2)', '(38, 48)', '(60, 100)', '(8, 12)'])

def preprocess_input(x):
    """
    Normalizes the image matrix to center the data at 0.
    The data is now between -0.5 ans 0.5
    :param x: the array of all instances
    :return: the array of all normalized instances
    """
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def generate_dataset(path='men_faces/train', mode='train'):

    while 1:
        with open('{}/{}_info.txt'.format(path, mode), 'r') as info:

            batch_step = 0

            X = np.empty([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
            Y = np.empty([BATCH_SIZE, NBR_OF_CLASSES])

            for line in info:
                # get image info and load image array
                img_name, gender, age = line.split(' ; ')
                img = load_img('{}/{}'.format(path, img_name), target_size=(IMAGE_SIZE, IMAGE_SIZE))
                x = img_to_array(img)
                X[batch_step] = x

                # get the age
                age_clean = age.strip()

                # encoded as a number
                encoded_age = encoder.transform([age_clean])
                # one hot encoded
                y = np_utils.to_categorical(encoded_age, num_classes=8)

                Y[batch_step] = y[0]
                batch_step += 1

                if batch_step == BATCH_SIZE:
                    yield preprocess_input(X), Y
                    batch_step = 0
                    X = np.empty([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
                    Y = np.empty([BATCH_SIZE, NBR_OF_CLASSES])


def base_model_training():
    # Using VGG face as the base model (only using the conv and pooling layers with pre-trained weights)
    vgg_model = VGGFace(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Don't train any of the layers yet, just train the FC layers we add
    for layer in vgg_model.layers:
        layer.trainable = False

    # The last layer we want is the last pooling layer before the FC layers
    last_layer = vgg_model.get_layer('pool5').output

    # Flattening before the FC
    last_layer = Flatten()(last_layer)

    # Adding two FC layers
    fc = Dense(4096, activation='relu')(last_layer)
    fc = Dense(2048, activation='relu')(fc)

    # Layer for the output probabilities
    output = Dense(8, activation='softmax', name='lb_output')(fc)

    model = Model(inputs=vgg_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Recording when the model is best regarding the validation accuracy
    filepath = "./weights/weights-improvement-{epoch:02d}--{val_acc:02f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    tensorboard = TensorBoard(log_dir='{}/logs/{}'.format(".", time()))

    model.fit_generator(generate_dataset(), steps_per_epoch=STEP_PER_EPOCH, epochs=EPOCH,
                        validation_data=generate_dataset(path='men_faces/valid', mode="valid"),
                        validation_steps=VALIDATION_SET_SIZE // BATCH_SIZE, callbacks=[tensorboard, checkpoint])


base_model_training()
