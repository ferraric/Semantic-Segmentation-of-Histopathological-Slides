import argparse
import os
from shutil import copyfile


def split_dataset_into_train_and_validation(
    path_to_dataset: str,
    train_output_folder_path: str,
    validation_output_folder_path: str,
    train_set_percentage: float,
):
    assert isinstance(
        path_to_dataset, str
    ), "path_to_dataset should be of type string but is of type {}".format(
        type(path_to_dataset)
    )
    assert isinstance(
        train_output_folder_path, str
    ), "train_output_folder_path should be of type string but is of type {}".format(
        type(path_to_dataset)
    )
    assert isinstance(
        validation_output_folder_path, str
    ), "validation_output_folder_path should be of type string but is of type {}".format(
        type(path_to_dataset)
    )
    assert isinstance(
        train_set_percentage, float
    ), "path_to_dataset should be of type float but is of type {}".format(
        type(train_set_percentage)
    )

    all_folder_elements = os.listdir(path_to_dataset)
    all_folder_elements.sort()
    total_slide_count = len(all_folder_elements) / 2
    slide_name = all_folder_elements[0].split("_")[0]
    number_of_individual_slides = 0
    for file_name in all_folder_elements:
        if file_name.split("_")[0] != slide_name:
            slide_name = file_name.split("_")[0]
            number_of_individual_slides += 1
    print(
        "We have {} patches and {} individual slides".format(
            total_slide_count, number_of_individual_slides
        )
    )

    number_of_individual_slides_for_train_set = int(
        train_set_percentage * number_of_individual_slides
    )
    train_file_paths = []
    validation_file_paths = []
    individual_slide_count = 0
    slide_name = all_folder_elements[0].split("_")[0]
    for file_name in all_folder_elements:
        if file_name.split("_")[0] != slide_name:
            individual_slide_count += 1
            slide_name = file_name.split("_")[0]

        if individual_slide_count < number_of_individual_slides_for_train_set:
            train_file_paths.append(os.path.join(path_to_dataset, file_name))
        else:
            validation_file_paths.append(os.path.join(path_to_dataset, file_name))
    print(
        "We have {} patches in the train set and {} patches in the validation set".format(len(train_file_paths) / 2, len(validation_file_paths) / 2))

    class_counter = {
        "background": 0,
        "spongiosis": 0,
        "epidermis": 0,
        "other": 0,
    }
    _count_number_of_classes(train_file_paths, class_counter)
    print("The training set has the following class counts: ")
    print(class_counter)

    class_counter = {
        "background": 0,
        "spongiosis": 0,
        "epidermis": 0,
        "other": 0,
    }
    _count_number_of_classes(validation_file_paths, class_counter)
    print("The validation set has the following class counts: ")
    print(class_counter)

    print("copying training")
    _copy_files_to_folder(train_file_paths, train_output_folder_path)
    print("copying validation")
    _copy_files_to_folder(validation_file_paths, validation_output_folder_path)

def _count_number_of_classes(folder_elements: list, class_counter: dict):
    assert isinstance(folder_elements, list), type(folder_elements)
    assert isinstance(class_counter, dict), type(class_counter)
    for element_path in folder_elements:
        file_name = os.path.split(element_path)[1]
        # get class name and increment counter
        file_name_parts = file_name.split("_")
        if(file_name_parts[0] == ".DS"):
            continue
        class_counter[file_name_parts[2]] += 1


def _copy_files_to_folder(file_paths: list, output_folder_path):
    assert isinstance(file_paths, list), type(file_paths)
    assert isinstance(output_folder_path, str), type(output_folder_path)
    if os.path.isdir(output_folder_path):
        "Folder {} can't be created as it is already a folder"
        return
    os.mkdir(output_folder_path)
    for file_path in file_paths:
        assert os.path.isfile(file_path)
        new_file_path = os.path.join(output_folder_path, os.path.split(file_path)[1])
        copyfile(file_path, new_file_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfolder", help="Add input folder path")
    argparser.add_argument(
        "-to", "--train_output_folder", help="Add train output folder path"
    )
    argparser.add_argument(
        "-vo", "--validation_output_folder", help="Add validation output folder path"
    )
    argparser.add_argument("-tp", "--train_percentage", help="Add train percentage")

    args = argparser.parse_args()
    split_dataset_into_train_and_validation(
        args.inputfolder,
        args.train_output_folder,
        args.validation_output_folder,
        float(args.train_percentage),
    )
