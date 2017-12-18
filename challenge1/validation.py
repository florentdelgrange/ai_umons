import progressbar
# see https://github.com/coagulant/progressbar-python3
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
import numpy as np
import os
import sys

class ArgumentsError(Exception):
    pass


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise ArgumentsError("Not enough arguments. Expected : model file name. Exemple:\n>> python3 validation.py model")

    model_name = sys.argv[1]
    # load json and create model
    json_file = open('{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("{}.h5".format(model_name))

    predictions = []

    length = file_len('sorted_faces/valid/valid_info.txt')

    bar = progressbar.ProgressBar(maxval=length, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    with open('sorted_faces/valid/valid_info.txt', 'r') as f:
        for i, line in enumerate(f):
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
            # print('{} : y={} | y_pred={} (F: {:.2f} %)'.format(len(predictions), gender, {True: 'm', False: 'f'}[p<0.5], p * 100))
            bar.update(i + 1)

    predictions = np.array(predictions)
    print('Accuracy : {:.2f} %'.format(len(predictions[predictions]) / len(predictions) * 100))
