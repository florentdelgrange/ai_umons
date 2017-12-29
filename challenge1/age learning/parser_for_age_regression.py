import glob
import os
import time
from shutil import copyfile

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def create_directories_men_age_regression():
    """
    Creates the directory structure for the men's age regression.
    """
    if not os.path.exists("men_faces"):
        os.makedirs("men_faces")

    if not os.path.exists("men_faces/all"):
        os.makedirs("men_faces/all")

    if not os.path.exists("men_faces/train"):
        os.makedirs("men_faces/train")

    if not os.path.exists("men_faces/valid"):
        os.makedirs("men_faces/valid")

    if not os.path.exists("weights_men_age_regression"):
        os.makedirs("weights_men_age_regression")


def parse_all_men(root_directory, training_size, make_bins=False, seed=42):
    """
    Goes trough a data description file and puts all the images into folders.
    All the images are renamed as a number. The folders contain a description file
    with the gender and age of the subject.

    Supposes there is a directory structure as follows :
     men_faces/
      all/
        all_info.txt
      train/
        train_info.txt
      valid/
        valid_info.txt


    :param root_directory: path to the openu folder
    :param training_size: size of the training set (in [0,1] as a percentage of the total number of samples)
    :param make_bins: change some of the age ranges to make more consistant bins
    :return:
    """

    # files with image information
    info_files = ["fold_frontal_0_data.txt", "fold_frontal_1_data.txt", "fold_frontal_2_data.txt",
                  "fold_frontal_3_data.txt", "fold_frontal_4_data.txt"]

    sep = "	"  # separation char

    count = 0  # counter for the lines

    output = "men_faces/"  # output for the image (and info file)

    all_info = open(output + "men_faces_info.txt", 'a')

    genders = []

    ages = []

    for file in info_files:

        with open(root_directory + file, 'r') as f:

            f.readline()  # skipping header

            for line in f:

                line = line.strip().split(sep)  # splitting the line

                # matching all file names for the current line (several different
                # names can be prepended to the name in the file)
                # 0 : folder, 2 : prepended name, 1 : filename + extension
                folder, prepended, filename = line[0], line[2], line[1]
                gender, age = line[4], line[3]
                for name in glob.glob(root_directory + "/faces/" + folder + '/*' + prepended + "." + filename):

                    # if the image is of a male and the age is in a range and its not an outlier like (8,12)
                    if gender == "m" and "," in age and age != "(8, 23)":

                        if make_bins:
                            if age in ["(38, 43)", "(38, 42)"]:
                                age = "(38, 48)"

                        # copy the file to the all/ folder
                        copyfile(name, output + "all/" + str(count) + ".jpg")
                        # filing the info file
                        all_info.write(str(count) + ".jpg" + " ; " + gender + " ; " + age + "\n")
                        ages.append(age)
                        genders.append(gender)

                        count += 1

    # close the info files
    all_info.close()

    time.sleep(1)

    np.random.seed(seed)

    skf = StratifiedShuffleSplit(n_splits=1, train_size=training_size, test_size=1 - training_size, random_state=seed)

    for train_index, valid_index in skf.split(np.zeros(count), np.array(ages)):

        train_info = open(output + "train/train_info.txt", 'a')

        for i in train_index:
            copyfile(output + "all/" + str(i) + ".jpg", output + "train/" + str(i) + ".jpg")

            train_info.write(str(i) + ".jpg" + " ; " + str(genders[i]) + " ; " + str(ages[i]) + "\n")

        train_info.close()

        valid_info = open(output + "valid/valid_info.txt", 'a')

        for i in valid_index:
            copyfile(output + "all/" + str(i) + ".jpg", output + "valid/" + str(i) + ".jpg")

            valid_info.write(str(i) + ".jpg" + " ; " + str(genders[i]) + " ; " + str(ages[i]) + "\n")

        valid_info.close()

        return len(train_index), len(valid_index)


create_directories_men_age_regression()

print(parse_all_men("/home/clement/Documents/drive/openu/", 0.75, make_bins=True))
