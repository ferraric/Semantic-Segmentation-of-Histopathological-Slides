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


import segmentation_models as sm
from utils.config import process_config

class PfalzTestdataLoader:
    def __init__(self, config, evaluation_folder_inputs):
        self.config = config
        self.preprocessing = sm.get_preprocessing(config.backbone)


        # create list of image and annotation paths
        self.slide_paths = []

        all_input_names = os.listdir(evaluation_folder_inputs)
        for input_name in all_input_names:
            self.slide_paths.append(os.path.join(evaluation_folder_inputs, input_name))

        self.slide_paths.sort()
        self.image_count = len(self.slide_paths)
        print("We found {} images".format(self.image_count))

        dataset = tf.data.Dataset.from_tensor_slices({
            'image_paths': self.slide_paths,
        })
        dataset = dataset.map(lambda x: (tf.py_function(self.parse_image_and_label, [x['image_paths']], [tf.float32, tf.string])))
        dataset = dataset.map(self._fixup_shape)

        self.dataset = dataset.repeat(1).batch(1, drop_remainder=False)

    def parse_image_and_label(self, image):
        image_path = image.numpy().decode('UTF-8')

        image_path_tensor = tf.io.read_file(image_path)
        img = tf.dtypes.cast(tf.image.decode_png(image_path_tensor, channels=3), tf.float32)

        assert img.shape == (self.config.image_size, self.config.image_size, 3), img.shape
        img = self.preprocessing(img)

        return img, image_path

    def _fixup_shape(self, images, path_names):
        images.set_shape([None, None, 3])
        return images, path_names

def evaluate_model_on_images(model, evaluation_folder_inputs, config, output_folder):
    assert os.path.isdir(evaluation_folder_inputs)
    print("Sanity checking that the model you are loading actually is the right one")
    #model.summary()

    number_of_test_images = 0


    test_data_set_loader = PfalzTestdataLoader(config, evaluation_folder_inputs)

    for i, el in enumerate(test_data_set_loader.dataset):
        print("Test image {} of {}".format(i+1, test_data_set_loader.image_count))
        logging.warning("Test image {} of {}".format(i+1, test_data_set_loader.image_count))
        input = el[0]
        image_path = el[1].numpy()[0].decode('UTF-8')

        image_name = os.path.split(image_path)[-1]
        prediction = "prediction" + model.predict(input)
        save_input_label_and_prediction(input, prediction, image_name, config, output_folder, i)

    assert number_of_test_images == test_data_set_loader.image_count, "Should be equal but is {} and {}".format(number_of_test_images, test_data_set_loader.image_count)


def save_input_label_and_prediction(input, prediction, image_name, config, output_folder, step ):

    assert input[0].numpy().shape == (config.image_size, config.image_size, 3), input[0].numpy().shape

    prediction_np = np.argmax(prediction[0], axis=-1).astype('uint8')
    assert prediction_np.shape == (config.image_size, config.image_size), prediction_np.shape
    prediction_image = Image.fromarray(prediction_np, "P")

    if(config.number_of_classes == 3):
        prediction_image.putpalette([
            255, 255, 255,  # white
            255, 0, 0,  # red
            0, 0, 255  # blue
        ])

    prediction_image.save(os.path.join(output_folder,image_name ))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-ei", "--evaluation_folder_inputs", help="Add folder with images you want to evaluate on")
    argparser.add_argument("-m", "--model_to_load", help="Add folder with the model you want to load")
    argparser.add_argument("-c", "--config", help="Pass some config, make sure to adjust the image size")
    argparser.add_argument("-o", "--output_folder", help="Output folder")
    logging.basicConfig(filename='results.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    args = argparser.parse_args()

    config = process_config(args.config)
    loaded_model = tf.keras.models.load_model(args.model_to_load, compile=False)
    evaluate_model_on_images(loaded_model, args.evaluation_folder_inputs, config, args.output_folder)