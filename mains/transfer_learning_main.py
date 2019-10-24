from comet_ml import Experiment
import tensorflow as tf
import json
import os
import random

from data_loader.general_data_loader import GeneralDataLoader
from models.transfer_learning_unet_model import TransferLearningUnet
from trainers.transfer_learning_unet_trainer import TransferLearningUnetTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # Create Comet logger
    experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )

    # create the experiments dirs and add the config file
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
    print("...creating foler {}".format(config.summary_dir))

    with open(
        os.path.join(config.summary_dir, "config_summary.json"), "w"
    ) as json_file:
        json.dump(config, json_file)

    # create your data generator and dump config file into it
    data = GeneralDataLoader(config, experiment)

    # # create a dummy model and feed it a random input data to get the architecture as summary
    # dummy_model = TransferLearningUnet(config)
    iterator = iter(data.test_data)
    dummy_inputs, b = next(iterator)
    a = 0
    #dummy_model(dummy_inputs)

    # model_architecture_path = os.path.join(config.summary_dir, "model_architecture")
    # with open(model_architecture_path, "w") as fh:
    #     # Pass the file handle in as a lambda function to make it callable
    #     dummy_model.summary(print_fn=lambda x: fh.write(x + "\n"))
    # # also lets print it once to the console
    # dummy_model.summary()
    # experiment.log_asset(model_architecture_path)
    #
    # # create an instance of the model you want
    # model = ExampleSegmentationModel(config)
    #
    # # create trainer and pass all the previous components to it
    # trainer = ExampleSegmentationTrainer(model, data, config, experiment)
    # # here you train your model
    # trainer.train()


if __name__ == "__main__":
    main()
