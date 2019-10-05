import numpy as np
import tensorflow as tf


class DataGenerator:
    def __init__(self, config, comet_logger):
        self.config = config
        self.comet_logger = comet_logger

        assert (
            "batch_size" in self.config
        ), "You need to define the parameter 'batch_size' in your config file."
        assert (
            "shuffle_buffer_size" in self.config
        ), "You need to define the parameter 'batch_size' in your config file."

        # load data here
        # lets take an input image that is all ones and whose pixels all belong to class 1
        # For the segmentation projects it makes sense to convert all labels to a bitmap of the same size as the image
        # with the elments indicating which class a pixel belongs to.
        self.inputs = np.zeros((1, 50, 50, 1))
        self.labels = np.ones((1, 50, 50, 1))
        # log the hash of the data. Not sure if we should use this as aour data is sensitive.
        # self.comet_logger.log_dataset_hash([self.inputs, self.labels])

        # Let's use tf dataset api
        self.train_data = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))
        self.train_data = self.train_data.shuffle(
            buffer_size=self.config.shuffle_buffer_size
        ).repeat(count=-1)
        self.train_data = self.train_data.batch(
            self.config.batch_size, drop_remainder=True
        )

        self.test_data = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))
        self.test_data = self.test_data.batch(self.config.batch_size)

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
