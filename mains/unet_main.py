from comet_ml import Experiment
import tensorflow as tf
import json
import os
import random
import json


from pathlib import Path
from data_loader.general_data_loader import GeneralDataLoader
from models.unet import UNetModel
from utils.config import process_config
from utils.dirs import create_dirs
from tensorflow.keras.metrics import MeanIoU, categorical_accuracy


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

    #Get Data
    data = GeneralDataLoader(config, experiment)



    model = UNetModel(config)
    model.compile(
        optimizer=config.optimizer,
        loss='categorical_crossentropy',
        metrics=[MeanIoU(3), categorical_accuracy],
        validation_data=data.test_data
    )

    checkpoint_path = config.model_save_path + "cp.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    if config.model_save_epoch > 0:
        model.load_weights(checkpoint_path)

    if config.model_save_epoch < config.num_epochs:
        with experiment.train():
            model.fit(data.train_data, 
                      epochs= config.num_epochs - config.model_save_epoch, 
                      verbose=1,
                      callbacks=[cp_callback, UpdateConfig()],
                      validation_data = data.test_data)

    with experiment.test():
        loss, iou, acc = model.evaluate(data.test_data)

        metrics = {
            'loss':loss,
            'MeanIoU': iou,
            'Accuracy': acc
        }
        experiment.log_metrics(metrics)



class UpdateConfig(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        config = process_config("configs/unet.json")
        config.model_save_epoch += 1
        json.dump(config, open("configs/unet.json", "w"), indent=2)


if __name__ == "__main__":
    main()
