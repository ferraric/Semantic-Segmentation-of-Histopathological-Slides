import argparse
import os, sys, inspect
from PIL import Image
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tools.oskal_etal_dataset_tools.image_slicer import image_slicer

def get_patch_metadata(dir):
    '''
    Returns a dict with key as the slide in question (or annotation) 
    and value a dict {'rows': x, 'columns': y} indicating how many rows and columns 
    of patches were extracted from the slide (or annotation)

    :param str dir: Directory where the files are present

    :return: Dictionary with format e.g. {'E_85': {'rows': x, 'columns': y}}
    :rtype: dict
    '''

    metadata = {}

    # Populating the Dictionary
    for file in os.listdir(dir):
        filename = file.split('.')[0]
        column = int(file[-9:-7])
        row = int(file[-6:-4])

        if filename in metadata:
            if metadata[filename]['rows'] < row:
                metadata[filename]['rows'] = row
            if metadata[filename]['columns'] < column:
                metadata[filename]['columns'] = column
        else:
            metadata[filename] = {'rows': row, 'columns': column}


    # Making the images Square
    for file in metadata:
        # If Larger Horizontally, remove columns on the right side of the image
        if metadata[file]['rows'] < metadata[file]['columns']:
            metadata[file]['columns'] = metadata[file]['rows']

        # If Larger Vertically, remove rows at the end
        elif metadata[file]['rows'] > metadata[file]['columns']:
            metadata[file]['rows'] = metadata[file]['columns']

    return metadata


def restitch_patches(dir):
    #dir = "C:\\Users\\guitb\\Documents\\LinuxWorkspace\\data\\segmentation\\test_annotations\\"
    metadata = get_patch_metadata(dir)
    patch_size = 4096

    for file_key in metadata:
        print(file_key)
        patches = []
        columns = []
        rows = []
        for file in os.listdir(dir):
            if file.split('.')[0] == file_key:
                column = int(file[-9:-7])
                row = int(file[-6:-4])
                
                if row > metadata[file_key]['rows'] or column > metadata[file_key]['columns']:
                    continue

                patches.append(Image.open(dir + file))
                columns.append((column-1) * patch_size)
                rows.append((row-1) * patch_size)

        stiched_img = Image.new('RGB', (max(rows) + patch_size, max(columns) + patch_size), None)
        for i in range(len(patches)):
            stiched_img.paste(patches[i], box=(rows[i], columns[i]))
            patches[i].close()
        stiched_img.resize((4096,4096)).save(dir + file_key +".png", "png")


#restitch_patches()
