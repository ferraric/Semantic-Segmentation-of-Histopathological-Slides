import numpy as np
from tools.helpers import find_nonzero_intervals, threshold


def compute_boxes(image, intensity_threshold=255):
    """Computes bounding boxes of nonwhite regions in image.

    Args:
        intensity_threshold (int): Value of intensity above which
            pixels are set to white.

    Returns: List of tuples [(x0, y0, x1, y1),...] of box coordinates
    """
    # Should thresholding be done here?
    image_thresholded = threshold(image, intensity_threshold)
    image_array = np.array(image_thresholded)
    boxes = []

    # Compute the number of white pixels for every columns and row
    if len(np.shape(image_array)) == 2:  # supposed to check which type, grayscale vs RGB etc
        whiteness_rows = np.sum(~image_array, axis=1).astype(int)
    else:
        # (white means (255, 255, 255), i.e. sum is 765)
        whiteness_rows = np.sum((np.sum(image_array, axis=2) != 765).astype(int), axis=1)

    nonwhite_intervals_rows = find_nonzero_intervals(whiteness_rows)

    for interval_rows in nonwhite_intervals_rows:
        image_cropped = image_thresholded.crop((0, interval_rows[0],
                                                image.size[0], interval_rows[1]))
        image_array_cropped = np.array(image_cropped)
        if len(np.shape(image_array)) == 2:
            whiteness_cols = np.sum(~image_array_cropped, axis=0).astype(int)
        else:
            whiteness_cols = np.sum((np.sum(image_array_cropped, axis=2) != 765).astype(int),
                                    axis=0)

        nonwhite_intervals_cols = find_nonzero_intervals(whiteness_cols)

        for interval_cols in nonwhite_intervals_cols:
            box = (interval_cols[0], interval_rows[0],
                   interval_cols[1], interval_rows[1])
            boxes += [box]

    return boxes


def compute_boxes_iteratively(image, iter_depth=2, intensity_threshold=254,
                              prune=True, pruning_threshold=.001):
    """Computes bounding boxes of nonwhite regions iteratively.

    This method calls the method compute_boxes() iteratively in order
    to separate nonwhite regions that could not be separated by
    previous calls.

    Args:

        iter_depth (int): Maximal number of iterations

        intensity_threshold (int): Value of intensity above which
            pixels are set to white.

        prune (bool): Specifies whether or not regions with area below a
            certain value are discarded

        pruning_threshold (float): Proportion of area of original
            image that a region needs to have in order not to be discarded

    Returns:
        List [(x0, y0, x1, y1), ...] of boxes found in the last iteration.
    """
    image_area = image.size[0] * image.size[1]

    _boxes = [[(0, 0, image.size[0], image.size[1])]]  # A list containing lists of boxes

    for i in range(iter_depth):
        _boxes_new = []
        for box in _boxes[-1]:
            _boxes = compute_boxes(image.crop(box), intensity_threshold)
            if prune:
                for i, _box in enumerate(_boxes):
                    if (_box[2] - _box[0])*(_box[3] - _box[1]) / image_area < pruning_threshold:
                        _boxes[i] = None
            _boxes_new += [(box[0] + _box[0],
                            box[1] + _box[1],
                            box[0] + _box[2],
                            box[1] + _box[3])
                           for _box in _boxes if _box is not None]
        _boxes += [_boxes_new]

    return _boxes[-1]
