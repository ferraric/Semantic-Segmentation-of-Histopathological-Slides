from comet_ml import Experiment
import tensorflow as tf
import json
import os
import random

from data_loader.general_data_loader import GeneralDataLoader
from models.unet import UNetModel
from utils.config import process_config
from utils.dirs import create_dirs


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    #try:
    #    args = get_args()
    #    config = process_config(args.config)

    #except:
    #    print("missing or invalid arguments")
    #    exit(0)

    config = process_config("configs/unet.json")

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
        os.path.join(config.summary_dir, "unet.json"), "w"
    ) as json_file:
        json.dump(config, json_file)

    model = UNetModel(config)


    #Get Data
    data = GeneralDataLoader(config, experiment)

    model.compile(
        optimizer=config.optimizer,
        loss='categorical_crossentropy',
    )

    model.fit(data.train_data, epochs=config.num_epochs, steps_per_epoch=config.batch_size)

    model.summary()
    model_architecture_path = os.path.join(config.summary_dir, "model_architecture")
    with open(model_architecture_path, "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + "\n"))
    experiment.log_asset(model_architecture_path)




if __name__ == "__main__":
    main()
