import glob
import os
from shutil import copyfile

dataset_base_dir = "sorted_gender_and_age"
age_categories = {
    "(0, 2)",
    "(4, 6)",
    "(8, 12)",
    "(13, 14)",
    "(15, 20)",
    "(21, 24)",
    "(25, 32)",
    "(33, 37)",
    "(38, 43)",
    "(44, 47)",
    "(48, 53)",
    "(54, 59)",
    "(60, 100)",
}


def create_directories():
    if not os.path.exists(dataset_base_dir):
        os.makedirs(dataset_base_dir)

    if not os.path.exists(dataset_base_dir + "/train"):
        os.makedirs(dataset_base_dir + "/train")

    if not os.path.exists(dataset_base_dir + "/valid"):
        os.makedirs(dataset_base_dir + "/valid")

    for age_category in age_categories:
        if not os.path.exists(dataset_base_dir + "/train/" + "f" + "." + age_category):
            os.makedirs(dataset_base_dir + "/train/" + "f" + "." + age_category)
        if not os.path.exists(dataset_base_dir + "/valid/" + "f" + "." + age_category):
            os.makedirs(dataset_base_dir + "/valid/" + "f" + "." + age_category)
        if not os.path.exists(dataset_base_dir + "/train/" + "m" + "." + age_category):
            os.makedirs(dataset_base_dir + "/train/" + "m" + "." + age_category)
        if not os.path.exists(dataset_base_dir + "/valid/" + "m" + "." + age_category):
            os.makedirs(dataset_base_dir + "/valid/" + "m" + "." + age_category)


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

        output = dataset_base_dir + "/train/"  # output for the image (and info file)

        for line in f:

            # selecting whether we add it to training or validation
            if count == training_nbr:
                # update the output path
                output = dataset_base_dir + "/valid/"

            line = line.split(sep)  # splitting the line

            is_valid = True

            # matching all file names for the current line (several different
            # names can be prepended to the name in the file)
            # 0 : folder, 2 : prepended name, 1 : filename + extension
            for name in glob.glob(root_directory + line[0] + '/*' + line[2] + "." + line[1]):
                age_data = line[3]
                gender_data = line[4]
                age_category = ""
                gender_category = ""

                # Age part

                try:
                    age = int(age_data)
                    # age is an int, assigning the image to the corresponding range
                    # Could be simplified but this is easier to read and extend if needed

                    if age < 0:
                        is_valid = False
                    elif 0 <= age <= 2:
                        age_category = "(0, 2)"
                    elif 4 <= age <= 6:
                        age_category = "(4, 6)"
                    elif 8 <= age <= 12:
                        age_category = "(8, 12)"
                    elif 13 <= age <= 14:
                        age_category = "(13, 14)"
                    elif 15 <= age <= 20:
                        age_category = "(15, 20)"
                    elif 21 <= age <= 24:
                        age_category = "(21, 24)"
                    elif 25 <= age <= 32:
                        age_category = "(25, 32)"
                    elif 33 <= age <= 37:
                        age_category = "(33, 37)"
                    elif 38 <= age <= 43:
                        age_category = "(38, 43)"
                    elif 44 <= age <= 47:
                        age_category = "(44, 47)"
                    elif 48 <= age <= 53:
                        age_category = "(48, 53)"
                    elif 54 <= age <= 59:
                        age_category = "(54, 59)"
                    elif 60 <= age <= 100:
                        age_category = "(60, 100)"
                    else:
                        is_valid = False

                except ValueError:
                    # age_data isn't an int so we'll assume it has to be a range
                    age_category = age_data
                    if age_category not in age_categories:
                        # Not one of the predefined ranges; ignoring that image (does not reject a lot of images)
                        # Some ranges in the dataset are hard to deal with as they overlap with others
                        is_valid = False

                # Gender part

                if gender_data == "f":
                    gender_category = "f"
                elif gender_data == "m":
                    gender_category = "m"
                else:
                    is_valid = False

                # We'll only use the image if it has been successfully classified for both age and gender
                # Obv introduces a bias as the assignment to the train/valid sets is done prior to that "filtering" step
                if is_valid:
                    copyfile(name, output + gender_category + "." + age_category + "/" + line[2] + "." + line[1])

                count += 1


# Quick check to make sure the dataset is there
if not os.path.exists("../openu"):
    print("openu dataset not found, please download it and put it in this project's root folder.")
    exit()

create_directories()
parse("../openu/fold_frontal_1_data.txt", "../openu/faces/", 0.75)
