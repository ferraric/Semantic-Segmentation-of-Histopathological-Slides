import openslide
from PIL.ImageDraw import Draw
from bboxes import compute_boxes_iteratively


class Slide(openslide.OpenSlide):
    """Extends openslide.OpenSlide with some convenience methods
    """

    def __init__(self, filename):
        super(Slide, self).__init__(filename)
        self.boxes = {}

    def compute_boxes(self, level=None, *args, **kwargs):
        """Computes bounding boxes for the slices on the slide.

        Args:
            level (int): The zoom level for which the boxes will be
               computed initially

            Other arguments can be passed that will be forwarded when
            calling computer_boxes_iteratively.

        After computing bounding box coordinates for the initial zoom
        level, they are rescaled to all other zoom levels. These
        coordinates are then stored in self.boxes, a dictionary whose
        keys are zoom levels, and with self.boxes[level] the
        coordinates of the bounding boxes at that zoom level.
        """
        if level is None:
            level = self.level_count - 1  # use thumbnail in lowest resolution by default
        size = self.level_dimensions[level]
        thumbnail = self.get_thumbnail(size)
        boxes = compute_boxes_iteratively(thumbnail, *args, **kwargs)
        self.boxes[level] = boxes

        for level in range(self.level_count):
            self.scale_boxes(level)

        return self.boxes

    def scale_boxes(self, level):
        """Scales bounding boxes to the desired zoom level.

        Args:
            level (int): zoom level to scale to

        If self.boxes has not been computed before, self.compute_boxes
        is called.
        """
        if len(self.boxes.keys()) == 0:
            self.compute_boxes
        if level not in self.boxes.keys():
            min_level = min(self.boxes.keys())
            boxes_new = []
            for box in self.boxes[min_level]:
                ratio = self.level_dimensions[level][0] / self.level_dimensions[min_level][0]
                box_new = (int(box[0] * ratio), int(box[1] * ratio),
                           int(box[2] * ratio), int(box[3] * ratio))
                boxes_new += [box_new]
            self.boxes[level] = boxes_new

    def get_thumbnail_with_boxes(self, level=None):
        """Gets a thumbnail with bounding boxes.

        Args:
            level (int): Zoom level of thumbnail

        Returns:
            thumbnail (PIL.Image.Image)
        """
        if level is None:
            level = self.level_count - 1

        if level not in self.boxes.keys():
            self.compute_boxes()

        thumbnail = self.get_thumbnail(self.level_dimensions[level])
        drawing = Draw(thumbnail)
        for box in self.boxes[level]:
            drawing.rectangle([box[0], box[1], box[2], box[3]],
                              outline='#ff0000')

        return thumbnail

    def get_region(self, box, level):
        """Gets a region specified by a box at a given zoom level.

        Args:
            box (tuple): Box (x0, y0, x1, y1) specifying the region 

            level (int): Zoom level

        Returns:
            region (PIL.Image.Image)
        """
        ratio = self.level_dimensions[0][0] / self.level_dimensions[level][0]
        x0 = int(box[0] * ratio)
        y0 = int(box[1] * ratio)
        width = box[2] - box[0]
        height = box[3] - box[1]
        region = self.read_region((x0, y0), level, (width, height))

        return region
