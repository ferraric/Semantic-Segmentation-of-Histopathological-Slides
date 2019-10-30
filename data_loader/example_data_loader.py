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
        self.inputs = np.zeros((1, 50, 50, 1))
        self.labels = np.ones((1, 50, 50, 1))

        self.train_data = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))
        self.train_data = self.train_data.shuffle(
            buffer_size=self.config.shuffle_buffer_size
        ).repeat(100)
        self.train_data = self.train_data.batch(
            self.config.batch_size, drop_remainder=True
        )

        self.validation_data = tf.data.Dataset.from_tensor_slices((self.inputs, self.labels))
        self.validation_data = self.validation_data.batch(self.config.batch_size)

        self.comet_logger.log_dataset_hash(self.train_data)
        self.comet_logger.log_dataset_hash(self.validation_data)
