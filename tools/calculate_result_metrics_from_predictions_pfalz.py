import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys, inspect
import argparse
import re
from PIL import Image
from itertools import groupby

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
Image.MAX_IMAGE_PIXELS = 100000000000

from utils.metrics import F1Score, MatthewsCorrelationCoefficient
import tensorflow.keras.metrics as tf_keras_metrics




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfolder", help="Add input folder path")
    args = argparser.parse_args()

    n_classes = 3

    accuracy = tf_keras_metrics.CategoricalAccuracy()
    mean_iou = tf.metrics.MeanIoU(num_classes=n_classes)
    matthews_correlation_coefficient = MatthewsCorrelationCoefficient(num_classes=n_classes)

    column_names = ["accuracy", "mean_iou", "matthews_correlation"]
    results = pd.DataFrame(columns = column_names)

    evaluation_folder = args.inputfolder

    for i, file_name in enumerate(os.listdir(evaluation_folder)):
        if ("label" in file_name):
            wsi_name = file_name.replace("label", "").replace(".png", "")
            print("looking at slice {}\n".format(wsi_name))
            prediction_name = file_name.replace("label", "prediction")
            
            label_np = np.array(Image.open(os.path.join(evaluation_folder, file_name)))
            label = tf.convert_to_tensor(label_np)
            label_one_hot = tf.dtypes.cast(tf.one_hot(label, n_classes), tf.float16)
            label = tf.dtypes.cast(label, tf.float16)
            prediction_np = np.array(Image.open(os.path.join(evaluation_folder, prediction_name)))
            prediction = tf.convert_to_tensor(prediction_np)
            prediction_one_hot = tf.dtypes.cast(tf.one_hot(prediction, n_classes), tf.float16)
            prediction = tf.dtypes.cast(prediction, tf.float16)

            accuracy.update_state(label_one_hot, prediction_one_hot)
            mean_iou.update_state(label, prediction)
            matthews_correlation_coefficient.update_state(label, prediction)

            results.loc[wsi_name] = {'accuracy': accuracy.result().numpy(),
                                     'mean_iou': mean_iou.result().numpy(),
                                     'matthews_correlation': matthews_correlation_coefficient.result().numpy()}
            
            accuracy.reset_states()
            mean_iou.reset_states()
            matthews_correlation_coefficient.reset_states()

    print("Results summary: \n")
    print(results)
    print("\n")
    print("Means: \n")
    print(results.mean(axis=0))
    print("\n")
    print("Medians: \n")
    print(results.median(axis=0))
    print("\n")
    print("Std deviations: \n")
    print(results.std(axis=0))
    results.to_pickle(os.path.join(evaluation_folder, "results"))

