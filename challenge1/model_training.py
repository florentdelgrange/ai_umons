import os
import sys
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from time import time
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

EPOCHS = 42
BATCH_SIZE = 32
STEPS_PER_EPOCH = 9146 // BATCH_SIZE
gender_dict = {'m': 0, 'f' : 1}
CUSTOM_SAVE_PATH = '.'

def save(model, path='.', model_name='xception_gender'):
    # serialize model to JSON
    model_json = model.to_json()
    with open("{}/models/{}.json".format(path, model_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}/models/{}.h5".format(path, model_name))
    print("Saved model to disk : {}".format(path))
    if path != '.':
        save(model, path='.', model_name=model_name)

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def generate_dataset(path='sorted_faces/train', mode='train', rotations=False):
    datagen_openu = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            #rotation_range=30,
            horizontal_flip=True,
            #fill_mode='nearest'
            #fill_mode='constant',
            #cval=0.,
            )

    while 1:
        with open('{}/{}_info.txt'.format(path, mode), 'r') as info:
            batch_step = 0
            # memory optimization
            X = np.empty([BATCH_SIZE, 299, 299, 3])
            Y = np.empty([BATCH_SIZE], dtype='uint8')
            for line in info:
                img_name, gender, age = line.split(' ; ')
                img = load_img('{}/all/{}'.format(path, img_name), target_size=(299, 299))
                x = img_to_array(img)
                X[batch_step] = x
                Y[batch_step] = gender_dict[gender]
                batch_step += 1

                if batch_step == BATCH_SIZE:
                    if rotations:
                        X, Y = datagen_openu.flow(x=X, y=Y, batch_size=BATCH_SIZE,
                                #save_to_dir='sorted_faces/gen'
                                ).next()
                    yield (preprocess_input(X), Y)
                    batch_step = 0
                    X = np.empty([BATCH_SIZE, 299, 299, 3])
                    Y = np.empty([BATCH_SIZE], dtype='uint8')

def fine_tuning(weights):
    # load json and create model
    json_file = open('{}/models/robust_xception_gender.json'.format(CUSTOM_SAVE_PATH), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("{}/models/robust_xception_gender.h5".format(CUSTOM_SAVE_PATH))

    if weights:
        model.load_weights(weights)

    # We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices
    for i, layer in enumerate(model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:55]:
       layer.trainable = False
    for layer in model.layers[55:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print("\n")
    print("Fine tuning phase")

    # Callbacks
    if not os.path.exists('{}/weights'.format(CUSTOM_SAVE_PATH)):
        os.makedirs("{}/weights".format(CUSTOM_SAVE_PATH))
    
    filepath= CUSTOM_SAVE_PATH + "/weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='{}/logs/{}'.format(CUSTOM_SAVE_PATH, time()))#, histogram_freq=1, write_grads=True, batch_size=BATCH_SIZE)

    # Fit
    model.fit_generator(generate_dataset(rotations=True), steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=generate_dataset(path='sorted_faces/valid', rotations=True),
                        validation_steps=15,
                        epochs=EPOCHS, callbacks=[tensorboard, checkpoint])
    save(model, path=CUSTOM_SAVE_PATH, model_name='fine_tuned_xception_gender')

def main_training(weights=''):
    print("model initialisation (Xception based) ...")
    base_model = Xception(include_top=False, input_shape=(299, 299, 3))

    # let's visualize layer names and layer indices
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add two fully-connected layer
    x = Dense(299, activation='relu')(x)
    # and a logistic layer ; we have 2 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:115]:
       layer.trainable = False
    for layer in model.layers[115:]:
       layer.trainable = True

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("\nTop layers fitting phase (with datagenerator)")

    # Callbacks
    if not os.path.exists('{}/weights'.format(CUSTOM_SAVE_PATH)):
        os.makedirs("{}/weights".format(CUSTOM_SAVE_PATH))

    filepath= CUSTOM_SAVE_PATH + "/weights/main_training_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='{}/logs/{}'.format(CUSTOM_SAVE_PATH, time()))#, histogram_freq=1, write_grads=True, batch_size=BATCH_SIZE)

    # Fit
    model.fit_generator(generate_dataset(rotations=True), steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=generate_dataset(path='sorted_faces/valid',
                                                         mode='valid', rotations=True),
                        validation_steps=15,
                        epochs=EPOCHS, callbacks=[tensorboard, checkpoint])

    # at this point, the top layers are well trained and we can start fine-tuning
    save(model, path=CUSTOM_SAVE_PATH, model_name='robust_xception_gender')

def model_initialisation_phase():
    print("model initialisation (Xception based) ...")
    base_model = Xception(include_top=False)

    # let's visualize layer names and layer indices
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(100, activation='relu')(x)
    # and a logistic layer ; we have 2 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    # Example : model.fit_generator(generate_arrays_from_file('/my_file.txt'),
    #                    steps_per_epoch=1000, epochs=10)
    print("\nTop layers fitting phase")

    tensorboard = TensorBoard(log_dir='{}/logs/{}'.format(CUSTOM_SAVE_PATH, time()))#, histogram_freq=1, write_grads=True, batch_size=BATCH_SIZE)

    model.fit_generator(generate_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS//2, callbacks=[tensorboard])
    #validation_data=generate_dataset('sorted_faces/valid', 'valid'), validation_steps=200)

    # at this point, the top layers are well trained and we can start fine-tuning
    save(model, CUSTOM_SAVE_PATH)


class TooManyArgumentsError(Exception):
    pass


if __name__ == '__main__':
    if not os.path.exists('{}/logs'.format(CUSTOM_SAVE_PATH)):
        os.makedirs("{}/logs".format(CUSTOM_SAVE_PATH))
    if not os.path.exists('{}/models'.format(CUSTOM_SAVE_PATH)):
        os.makedirs("{}/models".format(CUSTOM_SAVE_PATH))
    mode = ''
    weights = ''

    if len(sys.argv) > 1:
        if '--mode=' in ''.join(sys.argv[1:]):
            mode = list(map(lambda x: x.split('--mode=')[1],
                    filter(lambda s: s[:7]=='--mode=',
                        sys.argv[1:])))
            if len(mode) > 1:
                raise TooManyArgumentsError("Too many arguments with --mode option")
            mode = mode[0]

            weights_to_load = False
            for s in sys.argv[1:]:
                if weights_to_load:
                    weights = s
                    break
                elif s[:7] == '--mode=':
                    weights_to_load = True

    if mode == 'fine-tuning':
        fine_tuning(weights)
    elif mode == 'main-training':
        main_training(weights)
    else:
        model_initialisation_phase()
        main_training()
        fine_tuning()
