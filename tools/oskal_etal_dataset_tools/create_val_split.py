import argparse
import os
import random

def create_validation_split(train_image_folder, output_validaton_folder):
    if(os.path.isdir(output_validaton_folder)):
        print("path exists already!!")
        return
    else:
        os.mkdir(output_validaton_folder)
        os.mkdir(os.path.join(output_validaton_folder, "patches"))
        os.mkdir(os.path.join(output_validaton_folder, "label"))


    # first cound the number of patches and the different classes
    train_patches = os.listdir(os.path.join(train_image_folder, "patches"))
    train_labels = os.listdir(os.path.join(train_image_folder, "label"))


    assert len(train_patches) == len(train_labels), "should be same lenght but are {} and {}".format(len(train_patches), len(train_labels))
    number_of_train_patches = len(train_patches)
    number_of_background_patches = 0
    number_of_other_tissue_patches = 0
    number_of_eidermal_tissue_patches = 0

    background_files = []
    other_tissue_files = []
    epidermis_files = []

    for file in train_patches:
        class_name = os.path.splitext(file)[0][-6:]
        if("1" in class_name):
            number_of_background_patches += 1
            background_files.append(file)
        elif("2" in class_name):
            number_of_other_tissue_patches += 1
            other_tissue_files.append(file)
        elif("3" in class_name):
            number_of_eidermal_tissue_patches  += 1
            epidermis_files.append(file)

    print("number of total patches {}".format(number_of_train_patches))
    print("number of background {}".format(number_of_background_patches))
    print("number of other tissues {}".format(number_of_other_tissue_patches))
    print("number of epidermal {}".format(number_of_eidermal_tissue_patches))

    # create validation split
    random.shuffle(background_files)
    random.shuffle(other_tissue_files)
    random.shuffle(epidermis_files)
    validation_set_size = round(number_of_train_patches * 0.05)
    print("validation size {}".format(validation_set_size))

    current_validation_set_size = 0
    validation_files = []
    i = 0
    while(current_validation_set_size<=validation_set_size-3):
        validation_files.append(background_files[i])
        validation_files.append(other_tissue_files[i])
        validation_files.append(epidermis_files[i])
        i += 1
        current_validation_set_size += 3

    for validation_file in validation_files:
        source_patch = os.path.join(os.path.join(train_image_folder, "patches"), validation_file)
        destination_patch = os.path.join(os.path.join(output_validaton_folder, "patches"), validation_file)
        os.rename(source_patch, destination_patch)

        source_label = os.path.join(os.path.join(train_image_folder, "label"), validation_file)
        destination_label = os.path.join(os.path.join(output_validaton_folder, "label"), validation_file)
        os.rename(source_label, destination_label)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-t",
        "--train_image_folder",
        help="Add train_image_folder",
    )
    argparser.add_argument(
        "-o",
        "--output_validaiton_folder",
        help="Add output folder path",
    )
    args = argparser.parse_args()
    create_validation_split(
        args.train_image_folder, args.output_validaiton_folder)