from comet_ml import Experiment
import tensorflow as tf

import json
import random
import os,sys,inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import models.transfer_learning_models.transfer_learning_implementations as sm
sm.set_framework('tf.keras')
from data_loader.transfer_learning_data_generator import (
    TransferLearningDataLoader, NorwayTransferLearningDataLoader
)
from models.transfer_learning_models.transfer_learning_unet_model import (
    TransferLearningUnetModel,
)
from models.transfer_learning_models.transfer_learning_implementations.metrics import *
import tensorflow.keras.metrics as tf_keras_metrics
import tensorflow.keras.losses as tf_keras_losses
import models.transfer_learning_models.transfer_learning_implementations as sm

from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
    try:
        args = get_args()
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

    # define model and data
    transfer_learning_unet = TransferLearningUnetModel(config)
    model = transfer_learning_unet.model
    backbone_preprocessing = sm.get_preprocessing(config.backbone)
    if(config.norway_dataset):
        train_dataloader = NorwayTransferLearningDataLoader(
            config,
            validation=False,
            preprocessing=backbone_preprocessing
        )
        validation_dataloader = NorwayTransferLearningDataLoader(
            config,
            validation=True,
            preprocessing=backbone_preprocessing
        )
    else:
        train_dataloader = TransferLearningDataLoader(
            config,
            validation=False,
            preprocessing=backbone_preprocessing
        )
        validation_dataloader = TransferLearningDataLoader(
            config,
            validation=True,
            preprocessing=backbone_preprocessing
        )

    #print the model summary and save it into the output folder
    model.summary()
    model_architecture_path = os.path.join(config.summary_dir, "model_architecture")
    with open(model_architecture_path, "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + "\n"))
    experiment.log_asset(model_architecture_path)
    experiment.log_asset(args.config)


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=config.lr_decay_staircase)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    # Define metrics and losses
    iou_score = tf_keras_metrics.MeanIoU(num_classes=config.number_of_classes)
    precision = tf_keras_metrics.Precision()
    #recall = tf_keras_metrics.Recall()
    #f1_score = FScore(beta=1, class_indexes=config.number_of_classes)
    #f2_score = FScore(beta=2, threshold=0.5, class_indexes=config.number_of_classes)


    if(config.number_of_classes == 2):
        print("Binary Training")
        accuracy = tf_keras_metrics.BinaryAccuracy()
        loss = tf_keras_losses.CategoricalCrossentropy()
    else:
        accuracy = tf_keras_metrics.CategoricalAccuracy()
        loss = tf_keras_losses.CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[accuracy, tf_keras_metrics.Precision()],
    )

    #save_example_data(train_dataloader.dataset)




    model.fit(train_dataloader.dataset, epochs=config.num_epochs, steps_per_epoch=len(train_dataloader),
            validation_data=validation_dataloader.dataset,
            validation_steps=len(validation_dataloader),
            use_multiprocessing=False)


# def save_example_data(data_loader, comet_experiment=None):
#     for i, el in enumerate(data_loader):
#         tf.squeeze(el[1]).numpy()
#         im = Image.fromarray(tf.squeeze(el[1]).numpy(), 'P')
#         im.putpalette([
#             255, 255, 255, #white
#             255, 0, 0, #red
#             0,0,255  #blue
#         ])
#         im.save("/Users/jeremyscheurer/Desktop/label.png")
#         break


if __name__ == "__main__":
    main()
