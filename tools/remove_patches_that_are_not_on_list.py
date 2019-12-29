import os
import argparse
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-p", "--patches_path", help="Add patches folder path")
    argparser.add_argument("-c", "--correct_slides", help="Add correct_annotations file path")
    argparser.add_argument("-r", "--removed_folder", help="Add folder in which the removed images are placed")

    args = argparser.parse_args()
    os.mkdir(args.removed_folder)
    patches_path = args.patches_path
    all_patches = os.listdir(patches_path)
    correct_annotations = args.correct_slides
    with open(correct_annotations, "r") as f:
        all_lines = f.readlines()
    for patch_name in all_patches:
        found_slide_on_list = False
        for line in all_lines:
            if ("eMF" in line):
                slide_name = "eMF_" + line.split("_")[1] + "_"
            else:
                slide_name = line.split("_")[0] + "_"
            if(slide_name in patch_name):
                found_slide_on_list = True
                break
        if(not found_slide_on_list):
            os.rename(os.path.join(patches_path, patch_name), os.path.join(args.removed_folder, patch_name))




