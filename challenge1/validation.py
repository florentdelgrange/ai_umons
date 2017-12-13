from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from logger import Logger
import numpy as np
import os
import sys

DROPBOX_PATH = '.'

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':

    logs = Logger(filename='{}/accuracy.log'.format(DROPBOX_PATH))

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")

    predictions = []

    with open('sorted_faces/valid/valid_info.txt', 'r') as f:
        for line in f:
            name, gender, age = line.split(' ; ')
            img_path = 'sorted_faces/valid/all/{}'.format(name)
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            preds = model.predict(x)
            # decode the results into a list of tuples (class, description, probability)
            # (one such list for each sample in the batch)
            p = preds[0][0]
            predictions.append((p<0.5 and gender=='m') or (p>=0.5 and gender=='f'))
            print('{} : y={} | y_pred={} (F: {:.2f} %)'.format(len(predictions), gender, {True: 'm', False: 'f'}[p<0.5], p * 100))
    predictions = np.array(predictions)
    print('Accuracy : {:.2f} %'.format(len(predictions[predictions]) / len(predictions) * 100))
    
    logs.close()
