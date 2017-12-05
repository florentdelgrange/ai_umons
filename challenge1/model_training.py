from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
        for i in range():
            male_img = image.load_img('{}/men/{}'.format(path, i), target_size=(299, 299))
            female_img = image.load_img('{}/women/{}'.format(path, i), target_size=(299, 299))
            get_x = {}
            x = image.img_to_array(male_img)
            x = np.expand_dims(x, axis=0)
            get_x['male'] = x
            x = image.img_to_array(female_img)
            x = np.expand_dims(x, axis=0)
            get_x['female'] = x
            for genre in ['male', 'female']:
                for i, batch in enumerate(datagen.flow(get_x[genre], batch_size=1,
                    save_to_dir='sorted_faces/train/gen_faces', save_prefix='gen', save_format='jpeg')):
                    if i > 20:
                        break  # otherwise the generator would loop indefinitely
                    yield (batch, genre)


if __name__ == '__main__':
    base_model = InceptionResNetV2(include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer ; we have 2 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train the model on the new data for a few epochs
    # Example : model.fit_generator(generate_arrays_from_file('/my_file.txt'),
    #                    steps_per_epoch=1000, epochs=10)
    model.fit_generator(generate_dataset(), steps_per_epoch=1000, epochs=)

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
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(...)
