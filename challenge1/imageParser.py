from shutil import copyfile
import re
import glob
import os
import numpy as np


def create_directories():
    if not os.path.exists("sorted_faces"):
        os.makedirs("sorted_faces")

    if not os.path.exists("sorted_faces/train"):
        os.makedirs("sorted_faces/train")

    if not os.path.exists("sorted_faces/train/women"):
        os.makedirs("sorted_faces/train/women")

    if not os.path.exists("sorted_faces/train/men"):
        os.makedirs("sorted_faces/train/men")

    if not os.path.exists("sorted_faces/valid"):
        os.makedirs("sorted_faces/valid")

    if not os.path.exists("sorted_faces/valid/women"):
        os.makedirs("sorted_faces/valid/women")

    if not os.path.exists("sorted_faces/valid/men"):
        os.makedirs("sorted_faces/valid/men")

    if not os.path.exists("weights"):
        os.makedirs("weights")
        with open("weights/weights.h5", 'w'):
            pass
        with open("weights/weights_ft_tmp.h5", 'w'):
            pass
        with open("weights/weights_tl_tmp.h5", 'w'):
            pass


def create_directories_all():
    if not os.path.exists("sorted_faces"):
        os.makedirs("sorted_faces")

    if not os.path.exists("sorted_faces/train"):
        os.makedirs("sorted_faces/train")

    if not os.path.exists("sorted_faces/train/women"):
        os.makedirs("sorted_faces/train/women")

    if not os.path.exists("sorted_faces/train/men"):
        os.makedirs("sorted_faces/train/men")

    if not os.path.exists("sorted_faces/valid"):
        os.makedirs("sorted_faces/valid")

    if not os.path.exists("sorted_faces/valid/women"):
        os.makedirs("sorted_faces/valid/women")

    if not os.path.exists("sorted_faces/valid/men"):
        os.makedirs("sorted_faces/valid/men")

    if not os.path.exists("sorted_faces/all"):
        os.makedirs("sorted_faces/all")

    if not os.path.exists("sorted_faces/train/all"):
        os.makedirs("sorted_faces/train/all")

    if not os.path.exists("sorted_faces/valid/all"):
        os.makedirs("sorted_faces/valid/all")


def parse(filename, root_directory, training_size):
    """
    Goes trough a data description file and sorts the data into a training
    and validation set. Each set contains two folders, one containing men
    images and the other containing women images. Each folder contains a
    description file with the age of the subject.

    Supposes there is a directory structure as follows :
    sorted_faces/
      train/
        men/
        women/
        male_info.txt
        female_info.txt
      valid/
        men/
        women/
        male_info.txt
        female_info.txt

    :param filename: the description file
    :param root_directory : the path to the root directory containing the faces (faces/)
    :param training_size: size of the training set (in [0,1] as a percentage of the total number of samples)
    :return:
    """

    # nbr of lines
    nbr_lines = sum(1 for line in open(filename))

    sep = "	"  # separation char

    count = 0  # counter for the lines

    with open(filename, 'r') as f:

        f.readline()  # skipping header

        training_nbr = int(nbr_lines * training_size)  # nbr of training images

        output = "sorted_faces/"  # output for the image (and info file)

        female_info = open(output + "female_info.txt", 'a')

        male_info = open(output + "male_info.txt", 'a')

        for line in f:

            # selecting whether we add it to training or validation
            if count == training_nbr:
                # close the info files
                female_info.close()
                male_info.close()

                # update the output path
                output = "sorted_faces/valid/"

                # open new info files
                female_info = open(output + "women_info.txt", 'a')
                male_info = open(output + "male_info.txt", 'a')

            line = line.split(sep)  # splitting the line

            # matching all file names for the current line (several different
            # names can be prepended to the name in the file)
            # 0 : folder, 2 : prepended name, 1 : filename + extension
            for name in glob.glob(root_directory + line[0] + '/*' + line[2] + "." + line[1]):
                # 4 : male or female
                if line[4] == "f":
                    copyfile(name, output + "women/" + line[2] + "." + line[1])
                    female_info.write(line[2] + "." + line[1] + ";" + line[4] + ";" + line[3] + "\n")
                elif line[4] == "m":
                    copyfile(name, output + "men/" + line[2] + "." + line[1])
                    male_info.write(line[2] + "." + line[1] + ";" + line[4] + ";" + line[3] + "\n")

            count += 1
            # close the info files

        # close the info files
        female_info.close()
        male_info.close()


def parse_all(root_directory, training_size):
    """
    Goes trough a data description file and puts all the images in a single folder.
    All the images are renamed as a number. The folder contains a description file
    with the gender and age of the subject.

    Supposes there is a directory structure as follows :
     sorted_faces/
      all/
        all_info.txt
      train/
        men/
        women/
        all/
        train_info.txt
      valid/
        men/
        women/
        all/
        valid_info.txt


    :param filename: the description file
    :param root_directory : the path to the openu/ directory
    :param training_size: size of the training set (in [0,1] as a percentage of the total number of samples)
    :return:
    """

    # files with image information
    info_files = ["fold_frontal_0_data.txt", "fold_frontal_1_data.txt", "fold_frontal_2_data.txt",
                  "fold_frontal_3_data.txt", "fold_frontal_4_data.txt"]

    sep = "	"  # separation char

    count = 0  # counter for the lines

    output = "sorted_faces/"  # output for the image (and info file)

    all_info = open(output + "all_info.txt", 'a')

    genders = []

    ages = []

    for file in info_files:

        with open(root_directory + file, 'r') as f:

            f.readline()  # skipping header

            for line in f:

                line = line.split(sep)  # splitting the line

                # matching all file names for the current line (several different
                # names can be prepended to the name in the file)
                # 0 : folder, 2 : prepended name, 1 : filename + extension
                for name in glob.glob(root_directory + "/faces/" + line[0] + '/*' + line[2] + "." + line[1]):
                    copyfile(name, output + "all/" + str(count) + ".jpg")
                    all_info.write(str(count) + ".jpg" + " ; " + line[4] + " ; " + line[3] + "\n")
                    ages.append(line[3])
                    genders.append(line[4])

                    count += 1
                    # close the info files

    # close the info files
    all_info.close()

    import time
    time.sleep(2)

    shuffled_indices = np.random.permutation(count)

    print(count, len(ages))

    training_set_size = int(count * training_size)
    training_indices = shuffled_indices[:training_set_size]
    test_indices = shuffled_indices[training_set_size:]

    train_info = open(output + "train/train_info.txt", 'a')

    for i in training_indices:
        if str(genders[i]) == 'm':
            copyfile(output + "all/" + str(i) + ".jpg", output + "train/men/" + str(i) + ".jpg")
        elif str(genders[i]) == 'f':
            copyfile(output + "all/" + str(i) + ".jpg", output + "train/women/" + str(i) + ".jpg")

        copyfile(output + "all/" + str(i) + ".jpg", output + "train/all/" + str(i) + ".jpg")

        train_info.write(str(i) + ".jpg" + " ; " + str(genders[i]) + " ; " + str(ages[i]) + "\n")

    train_info.close()

    valid_info = open(output + "valid/valid_info.txt", 'a')

    for i in test_indices:
        if str(genders[i]) == "m":
            copyfile(output + "all/" + str(i) + ".jpg", output + "valid/men" + str(i) + ".jpg")
        elif str(genders[i]) == "f":
            copyfile(output + "all/" + str(i) + ".jpg", output + "valid/women" + str(i) + ".jpg")

        copyfile(output + "all/" + str(i) + ".jpg", output + "valid/all" + str(i) + ".jpg")

        valid_info.write(str(i) + ".jpg" + " ; " + str(genders[i]) + " ; " + str(ages[i]) + "\n")


create_directories_all()
parse_all("openu/", 0.75)
