import tensorflow as tf
from PIL import Image
import numpy as np
import os
import argparse
from utils.jeremy_metrics import  F1Score, MatthewsCorrelationCoefficient
import tensorflow.keras.metrics as tf_keras_metrics
import tensorflow.keras.losses as tf_keras_losses


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfolder", help="Add input folder path")
    args = argparser.parse_args()

    evaluation_folder = args.inputfolder
    count = 0

    accuracy = tf_keras_metrics.BinaryAccuracy()
    precision = tf_keras_metrics.Precision()  # positive predictive value in the paper
    recall = tf_keras_metrics.Recall()  # equivalent to sensitivity in the norway paper
    mean_iou = tf.metrics.MeanIoU(num_classes=2)
    f1_score = F1Score(num_classes=2, average='micro')  # dice similarity is equivalent to f1 score
    matthews_corelation_coefficient = MatthewsCorrelationCoefficient()

    average_acc = 0
    average_prec = 0
    average_recall = 0
    average_iou = 0
    average_f1 = 0
    average_matt = 0

    # metrics

    all_files = os.listdir(evaluation_folder)
    for file_name in all_files:
        if("label" in file_name):
            count += 1
            print("looking at file {}".format(file_name))
            index = file_name.split("_")[1]

            label = tf.dtypes.cast(tf.convert_to_tensor(np.array(Image.open(os.path.join(evaluation_folder, file_name)))), tf.float16)
            prediction = tf.dtypes.cast(tf.convert_to_tensor(np.array(Image.open(os.path.join(evaluation_folder, "prediction_" + index)))), tf.float16)

            accuracy.update_state(label, prediction)
            precision.update_state(label, prediction)
            recall.update_state(label, prediction)
            mean_iou.update_state(label, prediction)
            f1_score.update_state(label, prediction)
            matthews_corelation_coefficient.update_state(label, prediction)

            print(accuracy.result(), precision.result(), recall.result(), f1_score.result(), mean_iou.result(), matthews_corelation_coefficient.result())
            print("\n")

    print("accuracy", accuracy.result())
    print("recall", recall.result())
    print("precision", precision.result())
    print("mean iou", mean_iou.result())
    print("f1 score", f1_score.result())
    print("matthews", matthews_corelation_coefficient.result())
    print("count ", count)
