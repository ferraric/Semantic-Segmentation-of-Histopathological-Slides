import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D


class ExampleSegmentationModel(Model):
    """This is a simple example class that shows how to use the project architecture for semantic segmentation. """
    def __init__(self, config):
        super(ExampleSegmentationModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.conv1 = Conv2D(2, [2,2], padding="same", activation="relu")

    def call(self, x):
        x = self.conv1(x)
        return x
