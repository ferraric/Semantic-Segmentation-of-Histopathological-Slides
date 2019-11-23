import argparse
import os
from PIL import Image
import numpy as np
import scipy.misc as sci


def create_grond_truth_mask(epidermis_annotation_folder, foreground_annotation_folder, output_path):
    """
    Create a ground truth mask with 3 channels, where channel 0 = background, channel 1 = other tissue, channel 3 = epidermis
    :param epidermis_annotation_folder:
    :param foreground_annotation_folder:
    :return:
    """
    epidermis_annotation_files = os.listdir(epidermis_annotation_folder)
    foreground_annotation_files = os.listdir(foreground_annotation_folder)
    if(os.path.isdir(output_path)):
        print("Output folder already exists")
        return
    else:
        os.mkdir(output_path)

    assert len(epidermis_annotation_files) == len(foreground_annotation_files), \
        "lengths should be equal but are {} and {}".format(
        len(epidermis_annotation_files), len(foreground_annotation_files)
    )
    epidermis_annotation_files.sort()
    foreground_annotation_files.sort()

    for i, epidermis_file in enumerate(epidermis_annotation_files):
        print("creating GT for file {}".format(epidermis_file))
        assert epidermis_file.replace("epidermis", "") == foreground_annotation_files[i].replace("FG", "")
        epidermis = np.array(Image.open(os.path.join(epidermis_annotation_folder, epidermis_file)))
        foreground = np.array(Image.open(os.path.join(foreground_annotation_folder, foreground_annotation_files[i])))
        assert epidermis.shape == foreground.shape, "Should have same shap but have {} and {}".format(epidermis.shape, foreground.shape)
        (width, height) = epidermis.shape
        ground_truth = np.zeros((width, height, 3))
        ground_truth[:,:,2] = epidermis
        # get background
        ground_truth[:,:,0] = 1-foreground
        ground_truth[:,:,1] = np.logical_xor(foreground, epidermis)
        ground_truth = ground_truth.astype('uint8')

        ground_truth_image = Image.fromarray(ground_truth, mode="RGB")

        # Check that converting and loading doesnt change values.
        # ground_truth_image.save("/Users/jeremyscheurer/Desktop/gt.tif")
        # saved_im = np.array(Image.open("/Users/jeremyscheurer/Desktop/gt.tif"))
        #
        # for i in range(ground_truth.shape[0]):
        #     for j in range(ground_truth.shape[1]):
        #         for k in range(ground_truth.shape[2]):
        #             assert ground_truth[i,j,k] == saved_im[i,j,k],"At {},{},{} values are {} and {}".format(
        #                 i,j,k, ground_truth[i,j,k],saved_im[i,j,k]
        #             )
        ground_truth_image.save(os.path.join(output_path, os.path.splitext(epidermis_file)[0].replace("epidermis", "")) + "gt.tif")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-e",
        "--epidermis_annotation_folder",
        help="Add epiermis annotation folder path",
    )
    argparser.add_argument(
        "-f",
        "--foreground_annotation_folder",
        help="Add foreground annotation folder path",
    )
    argparser.add_argument(
        "-o",
        "--output_path",
        help="Add output folder path",
    )
    args = argparser.parse_args()
    create_grond_truth_mask(
        args.epidermis_annotation_folder, args.foreground_annotation_folder, args.output_path
    )
