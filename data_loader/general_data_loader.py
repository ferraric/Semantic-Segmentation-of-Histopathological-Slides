import tensorflow as tf
import pathlib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

class GeneralDataLoader:
    def __init__(self, config, comet_logger):
        self.config = config
        self.comet_logger = comet_logger

        assert (
            "batch_size" in self.config
        ), "You need to define the parameter 'batch_size' in your config file."
        assert (
            "shuffle_buffer_size" in self.config
        ), "You need to define the parameter 'batch_size' in your config file."
        assert (
                "dataset_path" in self.config
        ), "You need to define the parameter 'dataset_path' in your config file."


        # Note we are not actually going to load the whole images into our dataset, but only the paths to it and once
        # we get a batch we convert the path to an image
        data_dir = pathlib.Path(self.config.dataset_path)
        image_count = len(list(data_dir.glob('image*.png')))
        annotation_count = len(list(data_dir.glob('annotation*.png')))
        assert image_count == annotation_count, "The image count is {} and the annotation count is {}, but they should be" \
                                                "equal".format(image_count, annotation_count)

        list_input_paths = tf.data.Dataset.list_files(str(data_dir / 'image*'), shuffle=False)
        list_label_paths = tf.data.Dataset.list_files(str(data_dir / 'annotation*'), shuffle=False)

        input_dataset = list_input_paths.map(self.parse_image, num_parallel_calls=AUTOTUNE)
        label_dataset = list_label_paths.map(self.parse_image, num_parallel_calls=AUTOTUNE)
        self.train_data = tf.data.Dataset.zip((input_dataset, label_dataset))
        self.test_data = self.train_data
        self.test_data.batch(self.config.batch_size)


        self.train_data = self.train_data.shuffle(
            buffer_size=self.config.shuffle_buffer_size
        ).repeat(count=-1)
        self.train_data = self.train_data.batch(
            self.config.batch_size, drop_remainder=True
        )




    def parse_image(self, image_path):
        image_path_tensor = tf.io.read_file(image_path)
        # save imag as grayscale
        img = tf.image.decode_jpeg(image_path_tensor, channels=0)
        preprocessed_image = tf.image.convert_image_dtype(img, tf.float32)
        return preprocessed_image