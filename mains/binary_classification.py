from comet_ml import Experiment
import tensorflow as tf
import json
import random
import os, sys, inspect
import numpy as np
from PIL import Image

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data_loader.transfer_learning_data_generator import (
    BinaryClassificationDataloader
)
from models.transfer_learning_models.transfer_learning_unet_model import (
    TransferLearningUnetModel,
)
from utils.jeremy_metrics import MeanIouWithArgmax, F1Score, MatthewsCorrelationCoefficient
import tensorflow.keras.metrics as tf_keras_metrics
import tensorflow.keras.losses as tf_keras_losses

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    print("Let's Begin!")
    try:
        args = get_args()
        print(args.config)
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    if config.use_comet_experiments:
        experiment_id = experiment.connection.experiment_id
    else:
        experiment_id = "local" + str(random.randint(1, 1000000))

    config.summary_dir = os.path.join(
        "../experiments", os.path.join(config.exp_name, experiment_id), "summary/"
    )
    config.checkpoint_dir = os.path.join(
        "../experiments", os.path.join(config.exp_name, experiment_id), "checkpoint/"
    )
    create_dirs([config.summary_dir, config.checkpoint_dir])
    print("...creating folder {}".format(config.summary_dir))

    with open(
            os.path.join(config.summary_dir, "config_summary.json"), "w"
    ) as json_file:
        json.dump(config, json_file)

    # Define model and data
    ########################
    # load model
    print("loading model")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #model.summary()

    if (config.use_image_augmentations):
        print("using image augmentations")
        train_dataloader = BinaryClassificationDataloader(
            config,
            validation=False,
            preprocessing=None,
            use_image_augmentations=True
        )
    else:
        print("not using any imgae augmentations")
        train_dataloader = BinaryClassificationDataloader(
            config,
            validation=False,
            preprocessing=None,
            use_image_augmentations=False
        )

    validation_dataloader = BinaryClassificationDataloader(
        config,
        validation=True,
            preprocessing=None,
        use_image_augmentations=False
    )

    #print the model summary and save it into the output folder
    #model.summary()
    model_architecture_path = os.path.join(config.summary_dir, "model_architecture")
    #with open(model_architecture_path, "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        #model.summary(print_fn=lambda x: fh.write(x + "\n"))
    experiment.log_asset(model_architecture_path)
    experiment.log_asset(args.config)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=config.lr_decay_staircase)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Define metrics and losses
    ###########################
    print("Doing binary classification")
    loss = tf_keras_losses.binary_crossentropy
    accuracy = tf_keras_metrics.BinaryAccuracy()
    precision = tf_keras_metrics.Precision()  # positive predictive value in the paper
    recall = tf_keras_metrics.Recall()  # equivalent to sensitivity in the norway paper
    f1_score = F1Score(num_classes=2, average='micro', threshold=0.5)  # dice similarity is equivalent to f1 score
    matthews_corelation_coefficient = MatthewsCorrelationCoefficient()
    tp = tf.keras.metrics.TruePositives()
    tn = tf.keras.metrics.TrueNegatives()
    fn = tf.keras.metrics.FalseNegatives()
    fp = tf.keras.metrics.FalsePositives()
    metrics = [accuracy, precision, recall, f1_score, tp, tn, fn, fp]

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CosineSimilarity(),
        metrics=metrics
    )
    if (config.save_model):
        checkpoint_path = config.checkpoint_dir + "{epoch:02d}-{val_loss:.2f}.hdf5"
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           monitor='val_loss',
                                                           save_best_only=False,
                                                           save_weights_only=False,
                                                           save_freq='epoch',
                                                           verbose=1)
        callback = [save_callback]
        
        #model.build(tf.TensorShape([None, 4096, 4096, 6]))
        #model.load_weights("/cluster/home/beluis/experiments/transfer_learning_unet/d600a52ffd714b1da0ed496e2030ff6c/checkpoint/04-0.66.hdf5")
        
        class_weight = {0: (1 / 41)*(61)/2.0, 1: (1 / 20)*(61)/2.0}

        model.fit(train_dataloader.dataset, epochs=config.num_epochs, steps_per_epoch=len(train_dataloader),
                  validation_data=validation_dataloader.dataset,
                  validation_steps=len(validation_dataloader),
                  callbacks=callback,
                  #class_weight=class_weight,
                  use_multiprocessing=True)
    else:
        model.fit(train_dataloader.dataset, epochs=config.num_epochs, steps_per_epoch=len(train_dataloader),
                  validation_data=validation_dataloader.dataset,
                  validation_steps=len(validation_dataloader),
                  use_multiprocessing=True)



if __name__ == "__main__":
    main()
