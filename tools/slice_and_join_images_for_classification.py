import inspect,sys
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
grandparentdir = os.path.dirname(parentdir)

sys.path.insert(0,parentdir)
sys.path.insert(0,os.path.dirname(currentdir))

import argparse
from tools.oskal_etal_dataset_tools.image_slicer import image_slicer
from PIL import Image
import math

new_size = 4096
Image.MAX_IMAGE_PIXELS = 100000000000

def join_input_sclices(input_image_folder, output_image_folder):
    """Take a folder with many slices of many images and for each
    image recreate the original image from the slices"""
    all_input_image_names = os.listdir(input_image_folder)
    if (".DS_Store" in all_input_image_names):
        all_input_image_names.remove(".DS_Store")
    all_input_image_names.sort()
    previous_original_image_name = None
    all_slices_for_image = []
    total_slide_count = len(all_input_image_names)
    for i, image_name in enumerate(all_input_image_names):
        if("eMF" in image_name):
            current_slice_name = image_name.split("_")[0] + image_name.split("_")[1] + image_name.split("_")[2]
        else:
            current_slice_name = image_name.split("_")[0] + image_name.split("_")[1]
        current_slice_coordinates = ((int(image_name.split("_")[-1].split(".")[0])-1)*new_size,(int(image_name.split("_")[-2])-1)*new_size)
        if(i == 0):
            previous_original_image_name = current_slice_name
        if(previous_original_image_name == current_slice_name and i != total_slide_count-1):
            image_slice = Image.open(os.path.join(input_image_folder, image_name))
            tile = image_slicer.Tile(image=image_slice, number=0, coords=current_slice_coordinates, position=0)
            all_slices_for_image.append(tile)
        elif(i == total_slide_count-1 or previous_original_image_name != current_slice_name):
            print("Joining new image {}".format(previous_original_image_name))
            new_image = image_slicer.join(all_slices_for_image)
            new_image = new_image.resize((4096,4096), resample=Image.BICUBIC)
            new_image.save(os.path.join(output_image_folder, previous_original_image_name + ".png"))
            all_slices_for_image = []
            previous_original_image_name = current_slice_name
            image_slice = Image.open(os.path.join(input_image_folder, image_name))
            tile = image_slicer.Tile(image=image_slice, number=None, coords=current_slice_coordinates, position=None)
            all_slices_for_image.append(tile)


def split_input_images(input_image_folder, output_image_folder):
    """Take in a folder with images and split each image into slices of size 4096x4096 with first resizing
    the original image to the nearest multiple of 4096"""
    all_input_image_names = os.listdir(input_image_folder)
    if(".DS_Store" in all_input_image_names):
        all_input_image_names.remove(".DS_Store")
    print("Found {} images".format(len(all_input_image_names)))
    for i, image_name in enumerate(all_input_image_names):
        print("Splitting image {} of {}".format(i, len(all_input_image_names)))
        image_path = os.path.join(input_image_folder, image_name)
        image = Image.open(image_path)
        width, height = image.size
        print("Image {} has width {} and height {}".format(image_name, width, height))

        # downsample to nearest multiple of 4096
        downsample_width = math.floor(width / new_size)
        newWidth = int(downsample_width*new_size)
        downsample_height = math.floor(height / new_size)
        newHeight = int(downsample_height*new_size)
        print("resizing to powers of two: new_width= {} and new_height= {}".format( newWidth, newHeight))
        if(newWidth == 0 or newHeight == 0):
            continue
        image = image.resize((newWidth, newHeight), resample=Image.BICUBIC)

        # compute slices of size 4096x4096
        number_of_slice_columns = int(newWidth / new_size)
        number_of_slice_rows = int(newHeight / new_size)
        print("Generating {} sclices".format(number_of_slice_columns*number_of_slice_rows))
        image_tiles = image_slicer.slice(image, col=number_of_slice_columns, row=number_of_slice_rows, save=False)

        image_slicer.save_tiles(image_tiles, directory=output_image_folder, prefix=os.path.splitext(image_name)[0], format='png')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i",
        "--input_image_folder",
        help="Add test_image_folder",
    )
    argparser.add_argument(
        "-o",
        "--output_image_folder",
        help="Add output folder path",
    )
    argparser.add_argument(
        "-a",
        "--action",
        help="Action to perform, 0=split, 1=join",
    )
    args = argparser.parse_args()
    assert os.path.isdir(args.input_image_folder)
    if not os.path.isdir(args.output_image_folder):
        os.mkdir(args.output_image_folder)

    if int(args.action) == 0:
        split_input_images(args.input_image_folder, args.output_image_folder)

    elif int(args.action) == 1:
        join_input_sclices(args.input_image_folder, args.output_image_folder)

    else:
        raise NotImplementedError()
