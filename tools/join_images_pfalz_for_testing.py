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

def join_input_sclices(input_image_folder, output_image_folder, file_type_name):
    """Take a folder with many slices of many images and for each
    image recreate the original image from the slices"""
    all_input_image_names = os.listdir(input_image_folder)
    all_input_image_names.sort()
    all_input_image_names_copy = all_input_image_names.copy()
    for element in all_input_image_names_copy:
        if not file_type_name in element:
            all_input_image_names.remove(element)
    previous_original_image_name = None
    all_slices_for_image = []
    total_slide_count = len(all_input_image_names)
    for i, image_name in enumerate(all_input_image_names):
        if(file_type_name in image_name):
            current_slice_name = image_name.split("_")[1] + image_name.split("_")[2]
            print(current_slice_name)
            current_slice_coordinates = ((int(image_name.split("_")[-1].split(".")[0])-1)*new_size,(int(image_name.split("_")[-2])-1)*new_size)
            if(i == 0):
                previous_original_image_name = current_slice_name
            if(previous_original_image_name == current_slice_name and i != total_slide_count-1):
                image_slice = Image.open(os.path.join(input_image_folder, image_name))
                tile = image_slicer.Tile(image=image_slice, number=0, coords=current_slice_coordinates, position=0)
                all_slices_for_image.append(tile)
            elif(i == total_slide_count-1 or previous_original_image_name != current_slice_name):
                print("Joining new image {}".format(previous_original_image_name))
                new_image = image_slicer.join(all_slices_for_image, is_rgb=False)
                (new_width, new_height) = new_image.size
                new_image = new_image.resize((int(new_width/2),int(new_height/2)), resample=Image.BICUBIC)
                new_image.putpalette([
                    255, 255, 255,  # white
                    255, 0, 0,  # red
                    0, 0, 255  # blue
                ])
                new_image.save(os.path.join(output_image_folder, file_type_name + previous_original_image_name + ".png"))
                all_slices_for_image = []
                previous_original_image_name = current_slice_name
                image_slice = Image.open(os.path.join(input_image_folder, image_name))
                tile = image_slicer.Tile(image=image_slice, number=None, coords=current_slice_coordinates, position=None)
                all_slices_for_image.append(tile)




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

    args = argparser.parse_args()
    assert os.path.isdir(args.input_image_folder)
    if not os.path.isdir(args.output_image_folder):
        os.mkdir(args.output_image_folder)
    join_input_sclices(args.input_image_folder, args.output_image_folder, "label")
    join_input_sclices(args.input_image_folder, args.output_image_folder, "prediction")
