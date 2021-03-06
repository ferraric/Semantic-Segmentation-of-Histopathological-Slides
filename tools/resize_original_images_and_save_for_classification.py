import inspect,sys
import argparse
import os
from PIL import Image
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0,os.path.dirname(currentdir))

Image.MAX_IMAGE_PIXELS = 100000000000
new_size = 4096

def resize_and_save_images(input_image_folder, output_image_folder):
    """Takes images of an input folder, resizes them to new size and saves them with same name
    to an output folder"""
    all_input_image_names = os.listdir(input_image_folder)
    if (".DS_Store" in all_input_image_names):
        all_input_image_names.remove(".DS_Store")
    for image_name in all_input_image_names:
        image = Image.open(os.path.join(input_image_folder, image_name))
        
         # downsample to nearest multiple of 4096
        width, height = image.size
        downsample_width = math.floor(width / new_size)
        newWidth = int(downsample_width*new_size)
        downsample_height = math.floor(height / new_size)
        newHeight = int(downsample_height*new_size)
        print("resizing to powers of two: new_width= {} and new_height= {}".format( newWidth, newHeight))
        if(newWidth == 0 or newHeight == 0):
            continue
        image = image.resize((newWidth, newHeight), resample=Image.BICUBIC)
        
        
        resized_image = image.resize((new_size, new_size), resample=Image.BICUBIC)
        resized_image.save(os.path.join(output_image_folder, "resized"+image_name))

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
    resize_and_save_images(args.input_image_folder, args.output_image_folder)
