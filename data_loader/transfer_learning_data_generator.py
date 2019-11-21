import tensorflow as tf
import os
import math

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TransferLearningDataLoader:
    def __init__(self, config, validation=False, preprocessing=None, augmentation=None):
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.config = config
        if(validation):
            dataset_path = config.validation_dataset_path
        else:
            dataset_path = config.train_dataset_path

        # create list of image and annotation paths
        all_files = os.listdir(dataset_path)
        self.slide_paths = []
        self.annotation_paths = []
        for file in all_files:
            if "slide" in file:
                self.slide_paths.append(os.path.join(dataset_path, file))
            elif "annotation" in file:
                self.annotation_paths.append(os.path.join(dataset_path, file))

        self.slide_paths.sort()
        self.annotation_paths.sort()

        self.image_count = len(self.slide_paths)
        annotation_count = len(self.annotation_paths)
        assert self.image_count == annotation_count, (
            "The image count is {} and the annotation count is {}, but they should be"
            "equal".format(self.image_count, annotation_count)
        )
        for i, slide_path in enumerate(self.slide_paths):
            slide_name = os.path.split(slide_path)[1]
            annotation_name = os.path.split(self.annotation_paths[i])[1]
            assert slide_name.replace("slide", "") == annotation_name.replace(
                "annotation", ""
            ), (
                "Path names of slide {} and annotation {}"
                "do not match".format(slide_name, annotation_name)
            )

        print("We found {} images and annotations".format(self.image_count))

        dataset = tf.data.Dataset.from_tensor_slices((self.slide_paths, self.annotation_paths))
        dataset = dataset.map(self.parse_image_and_label, num_parallel_calls=AUTOTUNE)

        if(validation):
            self.dataset = dataset.repeat(-1).batch(self.config.batch_size, drop_remainder=True)
        else:
            self.dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size).repeat(-1).batch(self.config.batch_size, drop_remainder=True)

    def __len__(self):
        return math.ceil(self.image_count / self.config.batch_size)

    def parse_image_and_label(self, image_path, label_path):
        image_path_tensor = tf.io.read_file(image_path)
        label_path_tensor = tf.io.read_file(label_path)

        img = tf.image.decode_png(image_path_tensor, channels=3)
        label = tf.image.decode_png(label_path_tensor, channels=0)
        label = tf.dtypes.cast(tf.math.divide(label, 255), tf.uint8)

        if self.augmentation:
            img = self.augmentation(img)
            label = self.augmentation(label)

        if self.preprocessing:
            img = self.preprocessing(img)
        return img, label
