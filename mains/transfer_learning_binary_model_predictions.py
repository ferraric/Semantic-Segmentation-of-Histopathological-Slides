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


from utils.metrics import MeanIouWithArgmax, F1Score, MatthewsCorrelationCoefficient
import tensorflow.keras.metrics as tf_keras_metrics
import tensorflow.keras.losses as tf_keras_losses
import segmentation_models as sm
from utils.config import process_config
import efficientnet.tfkeras



class BinaryClassificationDataloader:
    def __init__(self, config, evaluation_folder_inputs):
        self.preprocessing = sm.get_preprocessing(config.backbone)
        self.config = config

        # create list of image and annotation paths
        all_files = os.listdir(evaluation_folder_inputs)
        self.slide_paths = []

        for file in all_files:
            if "mrxs" in file:
                self.slide_paths.append(os.path.join(evaluation_folder_inputs, file))


        self.slide_paths.sort()
        self.image_count = len(self.slide_paths)
        print("We found {} images".format(self.image_count))

        dataset = tf.data.Dataset.from_tensor_slices({
            'image_paths': self.slide_paths,
        })

        dataset = dataset.map(lambda x: (tf.py_function(self.parse_image_and_label, [x['image_paths']], [tf.float32, tf.uint8])))
        dataset = dataset.map(self._fixup_shape)

        self.dataset = dataset.repeat(1).batch(1, drop_remainder=False)


    def parse_image_and_label(self, image):
        image_path = image.numpy().decode('UTF-8')
        image_path_tensor = tf.io.read_file(image_path)
        img = tf.dtypes.cast(tf.image.decode_png(image_path_tensor, channels=3), tf.float32)
        img = tf.image.resize(img, (self.config.image_size, self.config.image_size),
                             method=tf.image.ResizeMethod.BILINEAR)
        if(os.path.split(image_path)[1][0]  == "E"):
            label = tf.keras.utils.to_categorical(0, num_classes=2)
        elif(os.path.split(image_path)[1][0] == "e"):
            label = tf.keras.utils.to_categorical(1, num_classes=2)

        label = tf.dtypes.cast(label, tf.uint8)

        if self.preprocessing:
            img = self.preprocessing(img)

        return img, label

    def _fixup_shape(self, images, labels):
        images.set_shape([self.config.image_size, self.config.image_size, 3])
        labels.set_shape([2])
        return images, labels

def evaluate_model_on_images(model, evaluation_folder_inputs, config):
    assert os.path.isdir(evaluation_folder_inputs)

    print("Sanity checking that the model you are loading actually is the right one")
    model.summary()

    accuracy = tf_keras_metrics.BinaryAccuracy()
    precision = tf_keras_metrics.Precision()  # positive predictive value in the paper
    recall = tf_keras_metrics.Recall()  # equivalent to sensitivity in the norway paper
    f1_score = F1Score(num_classes=2, average='micro', threshold=0.5)  # dice similarity is equivalent to f1 score
    matthews_corelation_coefficient = MatthewsCorrelationCoefficient()
    metrics = [accuracy, precision, recall, f1_score, matthews_corelation_coefficient]

    number_of_test_images = 0

    test_data_set_loader = BinaryClassificationDataloader(config, evaluation_folder_inputs)


    for i, el in enumerate(test_data_set_loader.dataset):
        print("Test image {} of {}".format(i+1, test_data_set_loader.image_count))
        logging.warning("Test image {} of {}".format(i+1, test_data_set_loader.image_count))
        input = el[0]
        label = el[1]

        prediction = model.predict(input)
        for metric in metrics:
            metric.update_state(label, prediction)

    assert number_of_test_images == test_data_set_loader.image_count, "Should be equal but is {} and {}".format(number_of_test_images, test_data_set_loader.image_count)
    print("number of test images", number_of_test_images)
    print("Accuracy : ", metrics[0].result())
    print("precision : ", metrics[1].result())
    print("recall : ", metrics[2].result())
    print("f1_score : ", metrics[3].result())
    print("matthews_corelation_coefficient : ", metrics[4].result())


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-ei", "--evaluation_folder_inputs", help="Add folder with images you want to evaluate on")
    argparser.add_argument("-m", "--model_to_load", help="Add folder with the model you want to load")
    argparser.add_argument("-c", "--config", help="Pass some config, make sure to adjust the image size")
    logging.basicConfig(filename='results.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    args = argparser.parse_args()

    config = process_config(args.config)
    loaded_model = tf.keras.models.load_model(args.model_to_load, compile=False)
    evaluate_model_on_images(loaded_model, args.evaluation_folder_inputs, config)