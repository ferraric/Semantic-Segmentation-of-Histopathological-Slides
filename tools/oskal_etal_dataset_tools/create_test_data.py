# resizing to nearest power of 2 is taken from https://gist.github.com/RandomEtc/241411
# image slicing is taken from https://github.com/samdobson/image_slicer
from tools.oskal_etal_dataset_tools.image_slicer import image_slicer

import argparse
import os
from PIL import Image
import math


def create_test_data(test_image_folder, test_annotation_folder, output_slices_folder, new_size):
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
    file = open("/Users/jeremyscheurer/Code/semantic-segmentation-of-histopathological-slides/tools/oskal_etal_dataset_tools/test_slides.txt")
    test_file_names = []
    for line in file:
        test_file_names.append(line[:-1])

    for test_image in all_images:
        if(test_image not in test_file_names):
            continue
        test_image_path = os.path.join(test_image_folder, test_image)
        annotation_image_name = os.path.splitext(test_image)[0]  + "_epidermis.png"
        annotation_image_path = os.path.join(test_annotation_folder, annotation_image_name)
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
                                prefix=os.path.splitext(test_image)[0], format='png')


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
    args = argparser.parse_args()
    create_test_data(
        args.test_image_folder, args.test_annotation_folder, args.output_slices_folder, int(args.resize_image_to))