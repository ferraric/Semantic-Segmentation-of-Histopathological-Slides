import tensorflow as tf
import argparse
import os, sys, inspect
import numpy as np
import math
import logging
from PIL import Image

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


import tensorflow.keras.losses as tf_keras_losses
import segmentation_models as sm
from utils.config import process_config
import efficientnet.tfkeras


class PfalzTestdataLoader:
    def __init__(self, config, evaluation_folder_inputs, evaluation_folder_labels):
        self.config = config
        self.preprocessing = sm.get_preprocessing(config.backbone)


        # create list of image and annotation paths
        self.slide_paths = []
        self.annotation_paths = []

        all_input_names = os.listdir(evaluation_folder_inputs)
        all_label_names = os.listdir(evaluation_folder_labels)
        for input_name in all_input_names:
            self.slide_paths.append(os.path.join(evaluation_folder_inputs, input_name))

        for label_name in all_label_names:
            self.annotation_paths.append(os.path.join(evaluation_folder_labels, label_name))

        self.slide_paths.sort()
        self.annotation_paths.sort()

        self.image_count = len(self.slide_paths)
        annotation_count = len(self.annotation_paths)
        assert self.image_count == annotation_count, (
            "The slide count is {} and the annotation count is {}, but they should be"
            " equal".format(self.image_count, annotation_count)
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
        dataset = dataset.map(lambda x: (tf.py_function(self.parse_image_and_label, [x['image_paths'], x['labels'], False], [tf.float32, tf.uint8])))
        dataset = dataset.map(self._fixup_shape)

        self.dataset = dataset.repeat(1).batch(1, drop_remainder=False)

    def parse_image_and_label(self, image, label, is_norwegian_data):
        image_path = image.numpy().decode('UTF-8')
        label_path = label.numpy().decode('UTF-8')

        image_path_tensor = tf.io.read_file(image_path)
        img = tf.dtypes.cast(tf.image.decode_png(image_path_tensor, channels=3), tf.float32)
        # Load image with Pillow to make sure we lod it in palette mode.        assert label.shape == (self.config.image_size, self.config.image_size, 1), label.shape
        label = np.expand_dims(np.array(Image.open(label_path)), -1).astype('uint8')

        if(is_norwegian_data):
            # somehow the anotations are loaded as 0 and 255 instead of 0 and 1, thus we just divide by 255
            if np.all(np.unique(label) == [0, 255]):
                label = np.divide(label, 255)

        assert label.shape[2] == 1, "label should have 1 channel but has {}".format(label.shape[2])
        label = tf.keras.utils.to_categorical(label, num_classes=self.config.number_of_classes)
        label = tf.dtypes.cast(label, tf.uint8)


        assert img.shape == (self.config.image_size, self.config.image_size, 3), img.shape
        assert label.shape == (self.config.image_size, self.config.image_size, self.config.number_of_classes), label.shape

        img = self.preprocessing(img)

        return img, label

    def _fixup_shape(self, images, labels):
        images.set_shape([None, None, 3])
        labels.set_shape([None, None, self.config.number_of_classes])
        return images, labels


class NorwayTestDataLoader(PfalzTestdataLoader):
    def __init__(self, config, evaluation_folder_inputs, evaluation_folder_labels):
        self.config = config
        self.preprocessing = sm.get_preprocessing(config.backbone)

        self.slide_paths = []
        self.annotation_paths = []

        all_input_names = os.listdir(evaluation_folder_inputs)
        all_label_names = os.listdir(evaluation_folder_labels)
        for input_name in all_input_names:
            self.slide_paths.append(os.path.join(evaluation_folder_inputs, input_name))

        for label_name in all_label_names:
            self.annotation_paths.append(os.path.join(evaluation_folder_labels, label_name))

        self.slide_paths.sort()
        self.annotation_paths.sort()

        self.image_count = len(self.slide_paths)
        annotation_count = len(self.annotation_paths)
        assert self.image_count == annotation_count, (
            "The slide count is {} and the annotation count is {}, but they should be"
            " equal".format(self.image_count, annotation_count)
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
            tf.py_function(self.parse_image_and_label, [x['image_paths'], x['labels'], True], [tf.float32, tf.uint8])))
        dataset = dataset.map(self._fixup_shape)

        self.dataset = dataset.repeat(1).batch(1, drop_remainder=False)



def evaluate_model_on_images(model, evaluation_folder_inputs, evaluation_folder_labels, config, output_folder):
    assert os.path.isdir(evaluation_folder_inputs)
    assert os.path.isdir(evaluation_folder_labels)

    print("Sanity checking that the model you are loading actually is the right one")
    model.summary()


    number_of_test_images = 0

    if(config.norway_dataset):
        test_data_set_loader = NorwayTestDataLoader(config, evaluation_folder_inputs, evaluation_folder_labels)
    else:
        test_data_set_loader = PfalzTestdataLoader(config, evaluation_folder_inputs, evaluation_folder_labels)

    for i, el in enumerate(test_data_set_loader.dataset):
        print("Test image {} of {}".format(i+1, test_data_set_loader.image_count))
        logging.warning("Test image {} of {}".format(i+1, test_data_set_loader.image_count))
        input = el[0]
        label = el[1]

        prediction = model.predict(input)
        save_input_label_and_prediction(input, label, prediction, config, output_folder, i)

    assert number_of_test_images == test_data_set_loader.image_count, "Should be equal but is {} and {}".format(number_of_test_images, test_data_set_loader.image_count)


def save_input_label_and_prediction(input, label, prediction, config, output_folder, step ):

    assert input[0].numpy().shape == (config.image_size, config.image_size, 3), input[0].numpy().shape
    assert label[0].numpy().shape == (config.image_size, config.image_size, config.number_of_classes), label[0].numpy().shape


    input_np = input[0].numpy().astype('uint8')
    assert input_np.shape == (config.image_size, config.image_size, 3), input_np.shape
    input_image = Image.fromarray(input_np, 'RGB')

    prediction_np = np.argmax(prediction[0], axis=-1).astype('uint8')
    assert prediction_np.shape == (config.image_size, config.image_size), prediction_np.shape
    prediction_image = Image.fromarray(prediction_np, "P")

    np_label = np.squeeze(np.argmax(label.numpy(), -1).astype('uint8'), axis=0)
    assert np_label.shape == (config.image_size, config.image_size), np_label.shape
    label_image = Image.fromarray(np_label, 'P')


    if(config.number_of_classes == 3):
        input_image.putpalette([
            255, 255, 255,  # white
            255, 0, 0,  # red
            0, 0, 255  # blue
        ])
        prediction_image.putpalette([
            255, 255, 255,  # white
            255, 0, 0,  # red
            0, 0, 255  # blue
        ])
        label_image.putpalette([
            255, 255, 255,  # white
            255, 0, 0,  # red
            0, 0, 255  # blue
        ])
    elif(config.number_of_classes ==2):
        prediction_image.putpalette([
            0, 0, 0,  # black
            255, 255, 255,  # white
        ])
        label_image.putpalette([
            0, 0, 0,  # black
            255, 255, 255,  # white
        ])

    else:
        raise NotImplementedError()

    input_image.save(os.path.join(output_folder, "input_{}.png".format(step)))
    prediction_image.save(os.path.join(output_folder, "prediction_{}.png".format(step)))
    label_image.save(os.path.join(output_folder, "label_{}.png".format(step)))



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-ei", "--evaluation_folder_inputs", help="Add folder with images you want to evaluate on")
    argparser.add_argument("-el", "--evaluation_folder_labels", help="Add folder with ground truth labels")
    argparser.add_argument("-m", "--model_to_load", help="Add folder with the model you want to load")
    argparser.add_argument("-c", "--config", help="Pass some config, make sure to adjust the image size")
    argparser.add_argument("-o", "--output_folder", help="Output folder")
    logging.basicConfig(filename='results.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    args = argparser.parse_args()

    config = process_config(args.config)
    loaded_model = tf.keras.models.load_model(args.model_to_load, compile=False)
    evaluate_model_on_images(loaded_model, args.evaluation_folder_inputs, args.evaluation_folder_labels, config, args.output_folder)