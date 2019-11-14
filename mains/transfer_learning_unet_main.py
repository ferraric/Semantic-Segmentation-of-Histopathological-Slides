from comet_ml import Experiment
import tensorflow as tf

SM_FRAMEWORK = tf.keras
import json
import os
import random
import sys
import numpy as np
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from data_loader.transfer_learning_data_generator import (
    TransferLearningData,
    TransferLearningGenerator,
    do_preprocessing,
    visualize,
)
from models.transfer_learning_models.transfer_learning_unet_model import (
    TransferLearningUnetModel,
)
from models.transfer_learning_models.transfer_learning_implementations.losses import *
from models.transfer_learning_models.transfer_learning_implementations.metrics import *
import tensorflow.keras.metrics as keras_metrics
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
    train_data = TransferLearningData(
        config.train_dataset_path,
        preprocessing=do_preprocessing(backbone_preprocessing),
    )
    validation_data = TransferLearningData(
        config.validation_dataset_path,
        preprocessing=do_preprocessing(backbone_preprocessing),
    )

    train_dataloader = TransferLearningGenerator(
        train_data, batch_size=config.batch_size, shuffle=True
    )
    validation_dataloader = TransferLearningGenerator(
        validation_data, batch_size=1, shuffle=False
    )

    # print the model summary and save it into the output folder
    model.summary()
    model_architecture_path = os.path.join(config.summary_dir, "model_architecture")
    with open(model_architecture_path, "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + "\n"))
    experiment.log_asset(model_architecture_path)

    model.compile(
        config.optimizer,
        loss=categorical_crossentropy,
        metrics=[keras_metrics.categorical_accuracy, iou_score, precision, recall, f2_score],
    )

    model.fit(train_dataloader, epochs=config.num_epochs, validation_data=validation_dataloader)


if __name__ == "__main__":
    main()
