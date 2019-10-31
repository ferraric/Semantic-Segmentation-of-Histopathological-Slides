from comet_ml import Experiment
import tensorflow as tf
import json
import os
import random
import sys

sys.path.append('../')

from data_loader.example_data_loader import DataGenerator
from models.example_segmentation_model import ExampleSegmentationModel
from trainers.example_segmentation_trainer import ExampleSegmentationTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.logger import log_model_architecture_to


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

    data = DataGenerator(config, experiment)

    model = ExampleSegmentationModel(config)
    data_input_shape = next(iter(data.train_data))[0].shape
    log_model_architecture_to(model, experiment, data_input_shape, config.summary_dir)

    trainer = ExampleSegmentationTrainer(model, data, config, experiment)
    trainer.train()


if __name__ == "__main__":
    main()
