import os
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.utils.np_utils import to_categorical
from logger import Logger

EPOCHS = 45
BATCH_SIZE = 2000
gender_dict = {'m': 0, 'f' : 1}
DROPBOX_PATH = '/home/florent/Dropbox/Info/ai_umons/challenge1'

def save(model, path='.'):
    # serialize model to JSON
    model_json = model.to_json()
    with open("{}/model.json".format(path), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}/model.h5".format(path))
    print("Saved model to disk")


def generate_dataset(path='sorted_faces/train'):
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    if not os.path.exists("sorted_faces/train/gen_faces"):
        os.makedirs("sorted_faces/train/gen_faces")
    while 1:
        with open('{}/train_info.txt'.format(path), 'r') as info:
            for line in info:
                img_name, gender, age = line.split(' ; ')
                img = load_img('{}/all/{}'.format(path, img_name), target_size=(299, 299))
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                #x = np.expand_dims(x, axis=0)
                for i, batch in enumerate(datagen.flow(x, batch_size=1,
                    save_to_dir='sorted_faces/train/gen_faces', save_prefix='gen', save_format='jpeg')):
                    if i > 10:
                        break # otherwise the generator would loop indefinitely
                    yield (batch, np.array([gender_dict[gender]]))

if __name__ == '__main__':
    logs = Logger(filename='{}/mylog.log'.format(DROPBOX_PATH))

    base_model = InceptionResNetV2(include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer ; we have 2 classes
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    # Example : model.fit_generator(generate_arrays_from_file('/my_file.txt'),
    #                    steps_per_epoch=1000, epochs=10)
    print("Top layers fitting phase")
    model.fit_generator(generate_dataset(), steps_per_epoch=BATCH_SIZE, epochs=EPOCHS)

    # at this point, the top layers are well trained and we can start fine-tuning
    # We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.011, momentum=0.9), loss='binary_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print("\n")
    print("Fine tuning phase")
    model.fit_generator(generate_dataset(), steps_per_epoch=BATCH_SIZE, epochs=EPOCHS)

    save(model)
    save(model, path=DROPBOX_PATH)

    logs.close()
