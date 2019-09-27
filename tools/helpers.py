import numpy as np
import PIL


def threshold(image, intensity_threshold=255):
    """Thresholds image based on intensity.

    Converts image to "L" mode (black and white) and sets all pixels
    above a specified intensity threshold to white and all others to
    blac.

    Args:
        intensity_threshold (int): Value of intensity above which
            pixels are set to white

    Returns: Thresholded image in "L" mode.

    """
    image_a = np.array(image.convert("L"))
    image_thresholded = img_frombytes((image_a >= intensity_threshold).astype("uint8"))

    return image_thresholded


def img_frombytes(data, mode="1"):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)

    return PIL.Image.frombytes(mode=mode, size=size, data=databytes)


def find_nonzero_intervals(array):
    """Finds nonzero intervals"""
    intervals = []
    current_interval = None  # None as long as in a zero region

    if array[0] != 0:
        current_interval = [0, 0]

    for i, value in enumerate(array[1:]):
        if current_interval is None and value != 0:
            current_interval = [i, 0]
        elif current_interval is not None and value == 0:
            current_interval[1] = i
            intervals += [current_interval]
            current_interval = None

    if current_interval is not None and current_interval[1] == 0:
        current_interval[1] = len(array)
        intervals += [current_interval]

    return intervals


def extract_red_part(image):
    """Creates a mask for extracting the red part (i.e. region of interest)
    """

    image_a = np.array(image.convert("HSV"))
    mask_a = np.logical_and(np.logical_or(image_a[:, :, 0] < 20,  # value
                                          image_a[:, :, 0] > 210),
                            image_a[:, :, 1] > 80,  # hue
                            image_a[:, :, 2] > 200).astype('uint8')  # saturation
    mask_im = img_frombytes(mask_a)
    white = PIL.Image.fromarray(np.tile(np.array([0, 0, 255]).astype('uint8'),
                                        np.shape(image_a)[0]*np.shape(image_a)[1]).
                                reshape(np.shape(image_a)), mode="HSV").convert("RGBA")
    composite = PIL.Image.composite(image, white, mask_im)

    return composite, mask_a
