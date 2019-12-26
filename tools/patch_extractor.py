import PIL
import numpy as np
import random
import argparse
import os
import time

from PIL import Image
from typing import Tuple, List, Dict
from scipy.spatial import distance

random.seed(18)

BACKGROUND = 0
EPIDERMIS = 1
SPONGIOSIS = 2
OTHER_TISSUE = 3


class PatchExtractor:
    def __init__(self, input_folder: str, output_folder: str, patch_size: Tuple[int, int], min_distance: int,
                 patches_per_class: List[int]):
        assert os.path.isdir(input_folder), "The given input path has to be a directory"
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.patch_size = patch_size
        self.min_distance_to_other_patches = min_distance
        self.valid_slide_and_annotation_names = []
        self.desired_number_of_patches_per_class = {BACKGROUND: patches_per_class[BACKGROUND],
                                                    EPIDERMIS: patches_per_class[EPIDERMIS],
                                                    SPONGIOSIS: patches_per_class[SPONGIOSIS],
                                                    OTHER_TISSUE: patches_per_class[OTHER_TISSUE]}
        self.allowed_number_of_consecutive_unsuccessful_iterations = 1
        Image.MAX_IMAGE_PIXELS = 100000000000

    def extract_and_save_patch_pairs(self):
        """
        This function loads images and annotation from self.input_folder. It randomly extracts matching pairs of patches
        from an image, it ensures that all patches have at least self.min_distance_to_other_patches (euclidean distance).
        Every extracted patch is evaluated and labeled as belonging to a certain class, the program runs until it has
        self.desired_number_of_patches_per_class or if it does not find a new patch for
        self.allowed_number_of_consecutive_unsuccessful_iterations.
        Lastly it saves the extracted patches to self.output_folder.
        """
        all_folder_elements = [f for f in os.listdir(self.input_folder) if not f.startswith('.')]
        for element in all_folder_elements:
            if ".mrxs" in element:
                found_comparison_element = False
                slide_name = element.split(".mrxs")[0]
                slide_index = os.path.splitext(element.split(".mrxs_")[1])[0]
                found_an_annotation = False
                for comparison_element in all_folder_elements:
                    if ("eMF" in comparison_element):
                        if (
                            "label" in comparison_element
                            and "eMF_" + comparison_element.split("_")[1] == slide_name
                                and os.path.splitext(comparison_element.split("_")[4])[0] == slide_index
                        ):
                            found_comparison_element = True

                    else:
                        if (
                            "label" in comparison_element
                            and comparison_element.split("_")[0] == slide_name
                            and os.path.splitext(comparison_element.split("_")[3])[0] == slide_index
                        ):
                            found_comparison_element = True

                    if(found_comparison_element):
                        found_an_annotation = True
                        self.valid_slide_and_annotation_names.append(
                            (element, comparison_element)
                        )
                        found_comparison_element = False
                        break
                assert (
                 found_an_annotation
                ), "We have not found an annotation for slide {}".format(slide_name)
        for (slide_name, annotation_name) in self.valid_slide_and_annotation_names:
            print(slide_name, annotation_name)
            sample_name = slide_name.split(".mrxs")[0]
            number_of_selected_patches_per_class = {BACKGROUND: 0,
                                                    EPIDERMIS: 0,
                                                    SPONGIOSIS: 0,
                                                    OTHER_TISSUE: 0}
            selected_top_left_corners = []
            annotation = Image.open(os.path.join(self.input_folder, annotation_name))
            slide = Image.open(os.path.join(self.input_folder, slide_name))
            consecutive_unsuccessful_iterations = 0
            patch_number = 0
            while not self.__enough_patches(number_of_selected_patches_per_class) \
                    and consecutive_unsuccessful_iterations < self.allowed_number_of_consecutive_unsuccessful_iterations:
                top_left_corner = self.__select_random_sufficiently_spaced_top_left_corner(
                    (annotation.height, annotation.width),
                    selected_top_left_corners)
                candidate_annotation_patch = self.__extract_patch_from(annotation, top_left_corner)
                slide_patch = self.__extract_patch_from(slide, top_left_corner)
                c = self.__determine_class_of(candidate_annotation_patch, slide_patch)
                if self.__class_still_needed(c, number_of_selected_patches_per_class):
                    print("useful patch found, class count: ", number_of_selected_patches_per_class)
                    number_of_selected_patches_per_class[c] += 1
                    selected_top_left_corners.append(top_left_corner)
                    slide_patch.save(os.path.join(self.output_folder,
                                                  sample_name + "_class_" + self.__class_to_string(c) + "_slide_"
                                                  + str(patch_number) + ".png"))
                    candidate_annotation_patch.save(os.path.join(self.output_folder,
                                                                 sample_name + "_class_" + self.__class_to_string(c)
                                                                 + "_annotation_" + str(patch_number) + ".png"),
                                                    mode='P')
                    patch_number += 1
                    consecutive_unsuccessful_iterations = 0
                else:
                    consecutive_unsuccessful_iterations += 1

    def __enough_patches(self, number_of_selected_patches_per_class: Dict[int, int]) -> bool:
        for c, count in number_of_selected_patches_per_class.items():
            if count < self.desired_number_of_patches_per_class[c]:
                return False
        return True

    def __select_random_sufficiently_spaced_top_left_corner(self,
                                                            im_shape: Tuple[int, int],
                                                            selected_top_left_corners: List[Tuple[int, int]]) \
            -> Tuple[int, int]:
        while True:
            candidate_top_left_corner = self.__select_random_valid_top_left_corner(im_shape)
            if self.__distance_to_existing_points_sufficient(candidate_top_left_corner, selected_top_left_corners):
                return candidate_top_left_corner

    def __select_random_valid_top_left_corner(self, im_shape: Tuple[int, int]) -> Tuple[int, int]:
        max_x = im_shape[0] - self.patch_size[0]
        max_y = im_shape[1] - self.patch_size[1]
        return (random.randint(0, max_x), random.randint(0, max_y))

    def __distance_to_existing_points_sufficient(self,
                                                 point: Tuple[int, int],
                                                 selected_top_left_corners: List[Tuple[int, int]]) -> bool:
        for existing_point in selected_top_left_corners:
            if distance.euclidean(point, existing_point) < self.min_distance_to_other_patches:
                return False
        return True

    def __extract_patch_from(self, im: PIL.Image, top_left_corner: Tuple[int, int]) \
            -> PIL.Image:
        return im.crop((top_left_corner[0], top_left_corner[1],
                        top_left_corner[0] + self.patch_size[0], top_left_corner[1] + self.patch_size[1]))

    def __determine_class_of(self, annotation_patch: Image, slide_patch: Image) -> int:
        pixel_values = np.array(annotation_patch).flatten()
        assert np.issubdtype(pixel_values.dtype, np.integer), \
            "annotation patch pixel values are assumed to be integers but were not"
        class_counts = np.bincount(pixel_values, minlength=3)
        class_percentages = np.array(list(map(lambda count: count / np.sum(class_counts), class_counts)))
        if class_percentages[SPONGIOSIS] > 0.2:
            return SPONGIOSIS
        if class_percentages[EPIDERMIS] + class_percentages[SPONGIOSIS] > 0.4:
            return EPIDERMIS
        else:
            if self.__is_background_patch(slide_patch):
                return BACKGROUND
            else:
                return OTHER_TISSUE

    def __is_background_patch(self, im: Image) -> bool:
        allowed_intensity_offset_from_max = 30
        allowed_number_of_outlier_pixels_per_intensity = 1000
        number_of_pixels_per_intensity = np.array(im.convert("L").histogram())
        lowest_non_outlier_intensity = np.argmax(
            number_of_pixels_per_intensity > allowed_number_of_outlier_pixels_per_intensity)
        return (255 - lowest_non_outlier_intensity < allowed_intensity_offset_from_max) \
               or self.__is_all_white(im)

    def __is_all_white(self, im: Image) -> bool:
        return im.convert("L").getextrema() == (0, 0)

    def __class_still_needed(self, c: int, number_of_selected_patches_per_class: Dict[int, int]) -> bool:
        return number_of_selected_patches_per_class[c] < self.desired_number_of_patches_per_class[c]

    def __class_to_string(self, c: int):
        if c == BACKGROUND:
            return "background"
        if c == EPIDERMIS:
            return "epidermis"
        if c == SPONGIOSIS:
            return "spongiosis"
        if c == OTHER_TISSUE:
            return "other_tissue"
        else:
            raise ValueError("{} does not correspond to a known class".format(c))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfolder", help="Add input folder path")
    argparser.add_argument("-o", "--outputfolder", help="Add output folder path")
    argparser.add_argument("-ps", "--patch_size", help="size of patches to extract in form width height", nargs=2,
                           type=int, default=[512, 512])
    argparser.add_argument("-md", "--min_distance", help="minimum euclidean pixel distance between patches", type=int,
                           default=200)
    argparser.add_argument("-nppc", "--patches_per_class",
                           help="desired number of patches per class in form n_background n_epidermis n_spongiosis n_other",
                           nargs=4, type=int, default=[10, 10, 10, 10])
    args = argparser.parse_args()
    patch_extractor = PatchExtractor(args.inputfolder, args.outputfolder, args.patch_size, args.min_distance,
                                     args.patches_per_class)
    patch_extractor.extract_and_save_patch_pairs()
