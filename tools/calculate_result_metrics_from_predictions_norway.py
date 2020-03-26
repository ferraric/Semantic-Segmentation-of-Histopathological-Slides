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

    accuracy = tf_keras_metrics.BinaryAccuracy()
    precision = tf_keras_metrics.Precision()  # positive predictive value in the paper
    recall = tf_keras_metrics.Recall()  # equivalent to sensitivity in the norway paper
    f1_score = F1Score(num_classes=2, average='micro')  # dice similarity is equivalent to f1 score
    mean_iou = tf.metrics.MeanIoU(num_classes=2)
    matthews_correlation_coefficient = MatthewsCorrelationCoefficient(num_classes=2)

    column_names = ["accuracy", "precision", "recall", "f1_score", "mean_iou", "matthews_correlation"]
    results = pd.DataFrame(columns = column_names)

    evaluation_folder = args.inputfolder
    all_files = list(filter(lambda f: (not "input" in f) and (not "results" in f), os.listdir(evaluation_folder)))
    all_files.sort(key=lambda x: re.sub("prediction_|label_","",x))
    extract_wsi_name = lambda f: re.split('_[0-9][0-9]_[0-9][0-9].png', re.split('prediction_|label_', f)[1])[0]

    for i, (wsi_name, slice_names) in enumerate(groupby(all_files, extract_wsi_name)):
        print("looking at WSI number {}: {}\n".format(i, wsi_name))
        slice_names = list(slice_names)
        for file_name in slice_names:
            if ("label" in file_name):
                print("looking at slice {}\n".format(file_name))
                prediction_name = file_name.replace("label", "prediction")
            
                label_np = np.array(Image.open(os.path.join(evaluation_folder, file_name)))
                label = tf.dtypes.cast(tf.convert_to_tensor(label_np), tf.float16)
                prediction_np = np.array(Image.open(os.path.join(evaluation_folder, prediction_name)))
                prediction = tf.dtypes.cast(tf.convert_to_tensor(prediction_np), tf.float16)

                accuracy.update_state(label, prediction)
                precision.update_state(label, prediction)
                recall.update_state(label, prediction)
                f1_score.update_state(label, prediction)
                mean_iou.update_state(label, prediction)
                matthews_correlation_coefficient.update_state(label, prediction)

        results.loc[wsi_name] = {'accuracy': accuracy.result().numpy(),
                                 'precision': precision.result().numpy(),
                                 'recall': recall.result().numpy(),
                                 'f1_score': f1_score.result().numpy(),
                                 'mean_iou': mean_iou.result().numpy(),
                                 'matthews_correlation': matthews_correlation_coefficient.result().numpy()}
        accuracy.reset_states()
        precision.reset_states()
        recall.reset_states()
        f1_score.reset_states()
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


