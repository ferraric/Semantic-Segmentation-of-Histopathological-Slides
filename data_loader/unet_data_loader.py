import tensorflow as tf
import os
import math
import numpy as np
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE


class UnetDataLoader:
    def __init__(self, config, validation=False, preprocessing=None, use_image_augmentations=False):
        self.use_image_augmentations = use_image_augmentations
        self.preprocessing = preprocessing
        self.config = config
        if (validation):
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

        dataset = tf.data.Dataset.from_tensor_slices({
            'image_paths': self.slide_paths,
            'labels': self.annotation_paths
        })
        dataset = dataset.map(lambda x: (
            tf.py_function(self.parse_image_and_label, [x['image_paths'], x['labels'], False],
                           [tf.float32, tf.float32])))
        dataset = dataset.map(self._fixup_shape)

        if (validation):
            self.dataset = dataset.repeat(-1).batch(self.config.batch_size)
        else:
            self.dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size).repeat(-1).batch(
                self.config.batch_size)

    def __len__(self):
        return math.ceil(self.image_count / self.config.batch_size)

    def parse_image_and_label(self, image, label, is_norwegian_data):
        image_path = image.numpy().decode('UTF-8')
        label_path = label.numpy().decode('UTF-8')

        image_path_tensor = tf.io.read_file(image_path)
        img = tf.image.decode_png(image_path_tensor, channels=3)

        # Load image with Pillow to make sure we lod it in palette mode.
        label = np.expand_dims(np.array(Image.open(label_path)), -1)
        assert label.shape[2] == 1, "label should have 1 channel but has {}".format(label.shape[2])

        if (is_norwegian_data):
            # somehow the anotations are loaded as 0 and 255 instead of 0 and 1, thus we just divide by 255
            label = np.divide(label, 255)

        label = tf.keras.utils.to_categorical(label, num_classes=self.config.number_of_classes)

        assert img.shape[2] == 3, "image should have 3 channels but has {}".format(img.shape[2])
        assert label.shape[2] == self.config.number_of_classes, "label should have {} channels but has {}".format(
            self.config.number_of_classes, label.shape[2])
        img = tf.cast(img, tf.float32)
        label = tf.cast(label, tf.float32)

        if self.use_image_augmentations:
            n_rotations = np.random.choice(4)
            img = tf.image.rot90(img, n_rotations)
            label = tf.image.rot90(label, n_rotations)

            if (np.random.rand(1) > 0.5):
                img = tf.image.flip_left_right(img)
                label = tf.image.flip_left_right(label)
            if (np.random.rand(1) > 0.5):
                img = tf.image.flip_up_down(img)
                label = tf.image.flip_up_down(label)

        if self.preprocessing:
            img = self.preprocessing(img)

        return img, label

    def _fixup_shape(self, images, labels):
        images.set_shape([None, None, 3])
        labels.set_shape([None, None, self.config.number_of_classes])
        return images, labels


class NorwayUnetDataLoader(UnetDataLoader):
    def __init__(self, config, validation=False, preprocessing=None, use_image_augmentations=False):
        self.use_image_augmentations = use_image_augmentations
        self.preprocessing = preprocessing
        self.config = config
        if (validation):
            dataset_path = config.validation_dataset_path
            print("Validating on the path {}".format(dataset_path))

        else:
            dataset_path = config.train_dataset_path
            print("Training on the path {}".format(dataset_path))

        # create list of image and annotation paths
        self.slide_paths = []
        self.annotation_paths = []

        for slide_file in os.listdir(os.path.join(dataset_path, "patches")):
            self.slide_paths.append(os.path.join(os.path.join(dataset_path, "patches"), slide_file))
        for annotation_file in os.listdir(os.path.join(dataset_path, "annotations")):
            self.annotation_paths.append(os.path.join(os.path.join(dataset_path, "annotations"), annotation_file))

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
            assert slide_name.replace("image", "") == annotation_name.replace(
                "annotation", ""
            ), (
                "Path names of slide {} and annotation {}"
                "do not match".format(slide_name, annotation_name)
            )

        print("We found {} images and annotations".format(self.image_count))

        dataset = tf.data.Dataset.from_tensor_slices({
            'image_paths': self.slide_paths,
            'labels': self.annotation_paths
        })
        dataset = dataset.map(lambda x: (
            tf.py_function(self.parse_image_and_label, [x['image_paths'], x['labels'], True],
                           [tf.float32, tf.float32])))
        dataset = dataset.map(self._fixup_shape)

        if (validation):
            self.dataset = dataset.repeat(-1).batch(self.config.batch_size, drop_remainder=True)
        else:
            self.dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size).repeat(-1).batch(
                self.config.batch_size, drop_remainder=True)
