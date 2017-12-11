import os

import numpy as np
from colorama import init, Back
from keras.preprocessing import image

from inception_v3 import InceptionV3
from inception_v3 import preprocess_input
from gender_and_age_combined.model_training import add_new_last_layer

init()  # required (on Windows only?) by colorama

age_categories = [
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
]


def testing(weights_path="weights_gender_and_age/weights.h5", dataset_base_dir="sorted_gender_and_age"):
    dir_list = next(os.walk(dataset_base_dir + '/valid'))[1]

    classes = dir_list
    classes = np.sort(classes)
    nb_classes = len(classes)
    # Setup the inceptionV3 model, pretrained on ImageNet dataset, without the fully connected part.
    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    # Add a new fully connected layer at the top of the base model. The weights of this FC layer are random
    # so they need to be trained
    model = add_new_last_layer(base_model, nb_classes)
    # We have already trained our model, so we just need to load it
    model.load_weights(weights_path)
    # Here, instead of writing the path and load the model each time, we load our model one time and we make a loop
    # where we ask only for the image path every time. If we enter "stop", we exit the loop

    file_processed = 0
    f_count = 0  # tested
    success_f_count = 0  # successfully classified
    m_count = 0  # tested
    success_m_count = 0  # successfully classified
    one_off = 0  # almost got the age category right
    two_off = 0
    three_off = 0
    four_off = 0
    more_off = 0
    success_age_count = 0
    fully_correct = 0  # both age and gender are correct

    file_count = sum([len(files) for r, d, files in os.walk(dataset_base_dir + "/valid/")])

    offsets = dict()  # stores how close to the actual age the prediction was

    for combined_class in dir_list:
        for root, dirs, files in os.walk(dataset_base_dir + "/valid/" + combined_class):
            print("Number of items in " + combined_class + ": " + str(len(files)))
            for file in files:
                file_processed = file_processed + 1
                if file.lower().endswith('.jpg'):
                    img_path = dataset_base_dir + "/valid/" + combined_class + "/" + file
                    if os.path.isfile(img_path):
                        img = image.load_img(img_path, target_size=(299, 299))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)

                        preds = model.predict(x)
                        # decode the results into a list of tuples (class, description, probability)
                        # (one such list for each sample in the batch)
                        label = classes[np.argmax(preds)]
                        p = preds[0][np.argmax(preds)] * 100
                        gender_ok = False
                        age_ok = False

                        if 'f' in label:  # classified as female
                            f_count = f_count + 1
                            if 'f' in combined_class:  # actually a female
                                success_f_count = success_f_count + 1
                                gender_ok = True

                        elif 'm' in label:
                            m_count = m_count + 1
                            if 'm' in combined_class:
                                success_m_count = success_m_count + 1
                                gender_ok = True

                        expected = 100  # dummy values that should never be used
                        predicted = -100

                        # As the age ranges are ordered, we can use the indices to determine
                        # how close to the expected value the prediction was
                        for index, cat in enumerate(age_categories):
                            if cat in combined_class:
                                expected = index
                            if cat in label:
                                predicted = index

                        offset = expected - predicted

                        offsets[offset] = offsets.get(offset, 0) + 1

                        if offset == 0:
                            age_ok = True
                            success_age_count = success_age_count + 1
                        elif abs(offset) == 1:
                            one_off = one_off + 1
                        elif abs(offset) == 2:
                            two_off = two_off + 1
                        elif abs(offset) == 3:
                            three_off = three_off + 1
                        elif abs(offset) == 4:
                            four_off = four_off + 1
                        else:
                            print("worse than 4-off:" + str(offset))
                            more_off = more_off + 1

                        if not gender_ok or not age_ok:
                            print("[class-err] Exp: " + combined_class + ", Got: " + label + " (p=" + (
                                "%.2f" % p) + "%, img=" + file + ")")
                        else:
                            fully_correct = fully_correct + 1

                    else:
                        print("Error")

                # Prints current progress in case we're dealing with a large dataset
                if file_processed % 50 == 0:
                    print("..." + "%.2f" % (100 * file_processed / file_count) + " %")

    total_age_classifications = success_age_count + one_off + two_off + three_off + four_off + more_off

    print()
    print("=> Female Accuracy: " + str(100 * success_f_count / f_count) + " %")
    print("=> Male Accuracy: " + str(100 * success_m_count / m_count) + " %")
    print("=> Gender global accuracy: " + "%.2f" % (
        100 * (success_m_count + success_f_count) / (m_count + f_count)) + " %")
    print("=> Gender average accuracy (in case test sets aren't equally distributed): " + "%.2f" % (
        (100 * success_f_count / f_count + 100 * success_m_count / m_count) / 2) + " %")
    print()
    print("====================================")
    print()
    print("=> Age Accuracy: " + str(100 * success_age_count / total_age_classifications) + " %")
    print("=> 1-off: " + str(100 * one_off / total_age_classifications) + " %")
    print("=> 2-off: " + str(100 * two_off / total_age_classifications) + " %")
    print("=> 3-off: " + str(100 * three_off / total_age_classifications) + " %")
    print("=> 4-off: " + str(100 * four_off / total_age_classifications) + " %")
    print("=> worse: " + str(100 * more_off / total_age_classifications) + " %")
    print()
    # Crappy histogram to display the age classification results in full yolo mode
    for key in sorted(offsets):
        to_print = str(key) + ":\t"
        for i in range(0, offsets[key] // 2):
            to_print = to_print + Back.GREEN + '_' + Back.RESET
        to_print = to_print + ' (' + str(offsets[key]) + ')'
        print(to_print)
    print()
    print("====================================")
    print()
    print("=> Full classification accuracy: " + str(100 * fully_correct / total_age_classifications) + " %")

    # print("Image " + file + " wrongly classified as " + label + " (" + ("%.2f" % p) + "%)")
    # orig = cv2.imread(img_path)
    # cv2.putText(orig, "Label: {}, {:.2f}%".format(label, p), (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.55, (0, 0, 255), 2, cv2.LINE_8, False)
    # cv2.imshow("Classification error", orig)
    # cv2.waitKey(5000)


if __name__ == "__main__":
    testing(weights_path="weights_gender_and_age/weights.h5", dataset_base_dir="sorted_gender_and_age")
