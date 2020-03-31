import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
from tools.slide import Slide
from PIL import Image


class CreateImageSegmentationPair:
    def __init__(
        self,
        input_slides_folder,
        output_folder,
        boxes_threshold,
        level,
        text_file_with_file_names=None,
    ):
        # Make some assertions about the files in the folder etc.
        assert os.path.isdir(input_slides_folder), "The given input path has to be a directory"
        #assert not os.path.isdir(
        #    output_folder
        #), "The given output path is already a directory"
        if(os.path.isdir(output_folder)):
            print("this output folder already exists")


        self.input_folder = input_slides_folder
        self.output_folder = output_folder
        self.box_threshold = boxes_threshold
        self.level = level
        self.valid_slides = []

        all_folder_elements = os.listdir(input_slides_folder)
        filename_folder_elements = all_folder_elements
        if text_file_with_file_names != None:
            filename_folder_elements = []

            with open(text_file_with_file_names, "r") as f:
                for line in f:
                    filename_folder_elements.append(line.rstrip())

        for element in all_folder_elements:
            if os.path.splitext(element)[1] == ".mrxs":
                self.valid_slides.append(element)

        Image.MAX_IMAGE_PIXELS = 100000000000
        # create the output folder
        if(not os.path.isdir(output_folder)):
            os.mkdir(output_folder)

    def create_dataset_of_image_segmentation_pairs(self):
        for i, slide_name in enumerate(
            self.valid_slides
        ):
            print(
                "We are at iteration {} of {} at slide name: {}".format(
                    i, len(self.valid_slides), slide_name
                )
            )
            file_exists = False
            for file_name in os.listdir(self.output_folder):
                if ("eMF" in file_name):
                    if (
                            "eMF_" + file_name.split("_")[1] == slide_name
                    ):
                        file_exists = True
                else:
                    if (
                            file_name.split("_")[0] == slide_name
                    ):
                        file_exists = True

            if(file_exists):
                print("This slide name exists already...skipping")
                continue
            slide = Slide(os.path.join(self.input_folder, slide_name))

            # Find the correspondence of segmentation and the box of interest
            _ = slide.compute_boxes(intensity_threshold=self.box_threshold)
            for index, box in enumerate(slide.boxes[self.level]):
                print(
                    "     We are at box {} out of {} boxes".format(
                        index, len(slide.boxes[self.level])
                    )
                )


                box_of_interest = slide.boxes[self.level][index]
                slide_of_interest = slide.get_region(box_of_interest, self.level)
                slide_of_interest.save(os.path.join(
                            self.output_folder,
                            slide_name + "_{}".format(str(index)) + ".png",
                        )
                    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-is", "--inputslidesfolder", help="Add input folder path")
    argparser.add_argument("-o", "--outputfolder", help="Add output folder path")
    argparser.add_argument(
        "-t",
        "--thresholdofbox",
        help="Add threshold parameter used to get the boxes around the slides",
    )
    argparser.add_argument(
        "-l",
        "--level",
        help="Add level threshold to indicate the resolution you want to have in your dataset",
    )
    argparser.add_argument(
        "-n",
        "--file_names",
        help="Add a path to a text file that contains annotation file names. If this text file is given, the corresponding"
        "annotation file for a .mrxs file will only be looked up from within the names given by this file.",
    )
    args = argparser.parse_args()
    dataset_creator = CreateImageSegmentationPair(
        args.inputslidesfolder,
        args.outputfolder,
        int(args.thresholdofbox),
        int(args.level),
        args.file_names,
    )
    dataset_creator.create_dataset_of_image_segmentation_pairs()


