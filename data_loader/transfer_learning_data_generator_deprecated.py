# License: Most of the Code in here has been taken from the multiclass example of https://github.com/qubvel/segmentation_models
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import albumentations as A
import tensorflow as tf
import math

# classes for data loading and preprocessing
class TransferLearningData:
    def __init__(self, dataset_path, augmentation=None, preprocessing=None):

        # create list of image and annotation paths
        all_files = os.listdir(dataset_path)
        self.slide_paths = []
        self.annotation_paths = []
        for file in all_files:
            if("slide" in file):
                self.slide_paths.append(os.path.join(dataset_path, file))
            elif("annotation" in file):
                self.annotation_paths.append(os.path.join(dataset_path, file))

        self.slide_paths.sort()
        self.annotation_paths.sort()

        self.image_count = len(self.slide_paths)
        annotation_count = len(self.annotation_paths)
        assert self.image_count == annotation_count, (
            "The image count is {} and the annotation count is {}, but they should be"
            "equal".format(self.image_count, annotation_count)
        )
        print("We found {} images and annotations".format(self.image_count))

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image_name = os.path.split(self.slide_paths[i])[1]
        annotation_name = os.path.split(self.annotation_paths[i])[1]
        assert image_name.replace("slide", "") == annotation_name.replace("annotation", ""), "Path names of slide {} and annotation {}" \
                                                                                         "do not match".format(image_name, annotation_name)
        # read data
        image = np.array(Image.open(self.slide_paths[i]))[..., :-1]
        annotation = np.array(Image.open(self.annotation_paths[i]))
        annotation = tf.keras.utils.to_categorical(annotation, 3)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, annotation=annotation)
            image, annotation = sample["image"], sample["annotation"]

        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)

        yield image, annotation

    def __len__(self):
        return self.image_count


class TransferLearningGenerator(tf.keras.utils.Sequence):
    """Load data from TransferLearningData and form batches
       Args:
           dataset: instance of TransferLearningData class for image loading, augmentation and preprocessing.
           batch_size: Integer number of images in batch.
           shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data

        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.indexes) / self.batch_size)


    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()