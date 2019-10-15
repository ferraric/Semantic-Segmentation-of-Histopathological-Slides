import os
import argparse
from tools.slide import Slide
from PIL import Image


class CreateImageSegmentationPair:
    def __init__(self, input_folder, output_folder, boxes_threshold, level):
        # Make some assertions about the files in the folder etc.
        assert os.path.isdir(input_folder), "The given input path has to be a directory"
        assert not os.path.isdir(
            output_folder
        ), "The given output path is already a directory"

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.box_threshold = boxes_threshold
        self.level = level
        self.valid_slides_and_annotations = []

        all_folder_elements = os.listdir(input_folder)
        for element in all_folder_elements:
            if os.path.splitext(element)[1] == ".mrxs":
                # make sure the file has an annotation
                slide_name = os.path.splitext(element)[0]
                found_an_annotation = False
                for comparison_element in all_folder_elements:
                    if comparison_element.startswith(
                        slide_name
                    ) and comparison_element.endswith("label.png"):
                        found_an_annotation = True
                        # add a tuple of the  slide name and corresponding annotation name to a list with all .mrxs slides
                        self.valid_slides_and_annotations.append(
                            (element, comparison_element)
                        )
                assert (
                    found_an_annotation
                ), "We have not found an annotation for slide {}".format(slide_name)

        Image.MAX_IMAGE_PIXELS = 100000000000
        # create the output folder
        os.mkdir(output_folder)

    def create_dataset_of_image_segmentation_pairs(self):
        # Loop over all slides and do the following:
        # 1. Cut out the slide that actually has an annotation
        # 2. Cut out the Annotation for this slide.

        # Here is how we achieve that:
        # The Problem is that we use a thresholding algorithm to get the boxes around our slides, i.e. depending
        # on the the threshold the boxes will be different. So we don't know a priori, which box belongs to the annotation region.
        # What we do is loop through all boxes and apply this box to the annotation files. We look at the pixel distribution of that$
        # box on the annotation file ,and if we find that it is not white, i.e. we have some annotation, we have the correspondence
        #  of the annotation with this box and cut out both the image and annotation at this box.

        for i, (slide_name, annotation_name) in enumerate(self.valid_slides_and_annotations):
            print("We are at iteration {} of {} at slide name: {}".format(i, len(self.valid_slides_and_annotations), slide_name))
            slide = Slide(os.path.join(self.input_folder, slide_name))
            annotation = Image.open(os.path.join(self.input_folder, annotation_name))

            # Find the correspondence of segmentation and the box of interest
            _ = slide.compute_boxes(intensity_threshold=self.box_threshold)
            index_to_box_of_interest = 0
            max_red_and_blue = 0
            for index, box in enumerate(slide.boxes[self.level]):
                print("     We are at box {} out of {} boxes".format(index, len(slide.boxes[self.level])))
                cropped_annotation = annotation.crop(box)
                # resize the cropped_annotation to save compute time and calculate the pixel distribution
                resized_cropped_annotation = cropped_annotation.resize((100, 100))
                # the first channel is the white channel, the second and third indicate the blue and red color
                color_histogram = resized_cropped_annotation.histogram()[:3]
                red_and_blue = color_histogram[1] + color_histogram[2]
                if red_and_blue > max_red_and_blue:
                    max_red_and_blue = red_and_blue
                    index_to_box_of_interest = index

            # We know know the box of interest. Let's crop the image and annotation to that box and save the resulting images.
            # We didn't directly save the images and annotations already cropped before, because of memory
            box_of_interest = slide.boxes[self.level][index_to_box_of_interest]
            slide_of_interest = slide.get_region(box_of_interest, self.level)
            annotation_of_interest = annotation.crop(box_of_interest)
            slide_of_interest.save(
                os.path.join(self.output_folder, slide_name + ".png")
            )
            annotation_of_interest.save(
                os.path.join(self.output_folder, annotation_name)
            )





if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfolder", help="Add input folder path")
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
    args = argparser.parse_args()
    dataset_creator = CreateImageSegmentationPair(
        args.inputfolder, args.outputfolder, int(args.thresholdofbox), int(args.level)
    )
    dataset_creator.create_dataset_of_image_segmentation_pairs()
