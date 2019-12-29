# resizing to nearest power of 2 is taken from https://gist.github.com/RandomEtc/241411
# image slicing is taken from https://github.com/samdobson/image_slicer
import argparse
import os, sys, inspect
from PIL import Image
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tools.oskal_etal_dataset_tools.image_slicer import image_slicer



def create_test_data(test_image_folder, test_annotation_folder, output_slices_folder, new_size, file_with_slide_names):
    """Take a folder with WSI Test images, resize every image to be divisible by new_size and then create tiles of
    size new_sizexnew_size"""
    Image.MAX_IMAGE_PIXELS = 100000000000
    if(os.path.isdir(output_slices_folder)):
        print("Directory already exists")
        return
    else:
        os.mkdir(output_slices_folder)
        output_slide_folder = os.path.join(output_slices_folder, "slides")
        output_annotation_folder = os.path.join(output_slices_folder, "annotations")
        os.mkdir(output_slide_folder)
        os.mkdir(output_annotation_folder)
    print("new size {}".format(new_size))
    all_images = os.listdir(test_image_folder)
    file = open(file_with_slide_names)
    test_file_names = []
    for line in file:
        test_file_names.append(line[:-1])

    for test_image in all_images:
        is_in_test_file_names = False
        for test_file in test_file_names:
            if(test_image.split(".")[0]  + "_"  in test_file):
                is_in_test_file_names = True
                break
        if(not is_in_test_file_names):
            continue
        test_image_path = os.path.join(test_image_folder, test_image)
        # get annotation name
        test_image_name = test_image.split(".")[0]
        test_image_index = test_image.split(".")[1][-1]
        for annotation_file in os.listdir(test_annotation_folder):
            if(test_image_name + "_" in annotation_file and "labels_" + test_image_index + "." in annotation_file):
                annotation_image_path = os.path.join(test_annotation_folder, annotation_file)
                break

        image = Image.open(test_image_path)
        annotation = Image.open(annotation_image_path)
        width, height = image.size
        print("Image {} has width {} and height {}".format(test_image, width, height))
        wmulti = math.floor(width / new_size)
        newWidth = int(wmulti * new_size)
        hmulti = math.floor(height / new_size)
        newHeight = int(hmulti * new_size)
        print("resizing to powers of two: new_width= {} and new_height= {}".format( newWidth, newHeight))
        image = image.resize((newWidth, newHeight), resample=Image.BICUBIC)
        annotation = annotation.resize((newWidth, newHeight), resample=Image.BICUBIC)

        number_of_slice_columns = int(newWidth / new_size)
        number_of_slice_rows = int(newHeight / new_size)
        image_tiles = image_slicer.slice(image, col=number_of_slice_columns, row=number_of_slice_rows, save=False)
        annotation_tiles = image_slicer.slice(annotation, col=number_of_slice_columns, row=number_of_slice_rows, save=False)
        image_slicer.save_tiles(image_tiles, directory=output_slide_folder, \
                                prefix=os.path.splitext(test_image)[0], format='png')
        image_slicer.save_tiles(annotation_tiles, directory=output_annotation_folder, \
                                prefix=os.path.splitext(annotation_file)[0], format='png')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-t",
        "--test_image_folder",
        help="Add test_image_folder",
    )
    argparser.add_argument(
        "-a",
        "--test_annotation_folder",
        help="Add test_annotation_folder",
    )
    argparser.add_argument(
        "-o",
        "--output_slices_folder",
        help="Add output folder path",
    )
    argparser.add_argument(
        "-s",
        "--resize_image_to",
        help="Resize imag to s",
    )
    argparser.add_argument(
        "-f",
        "--file_with_slide_names",
        help="a file with names of slides"
    )
    args = argparser.parse_args()
    create_test_data(
        args.test_image_folder, args.test_annotation_folder, args.output_slices_folder, int(args.resize_image_to), args.file_with_slide_names)