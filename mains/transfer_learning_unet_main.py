from comet_ml import Experiment
import tensorflow as tf
import json
import random
import os,sys,inspect
import numpy as np
from PIL import Image
import albumentations as A

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from data_loader.transfer_learning_data_generator import (
    TransferLearningDataLoader, NorwayTransferLearningDataLoader
)
from models.transfer_learning_models.transfer_learning_unet_model import (
    TransferLearningUnetModel,
)
from models.unet import UNetModel
from utils.metrics import MeanIouWithArgmax, F1Score, MatthewsCorrelationCoefficient
import tensorflow.keras.metrics as tf_keras_metrics
import tensorflow.keras.losses as tf_keras_losses
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

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

    # Define model and data
    ########################
    model = UNetModel(config)
    #model = norway_unet_model.model
    #backbone_preprocessing = sm.get_preprocessing(config.backbone)

    if(config.norway_dataset):
        print("using norwegian dataset")
        assert config.number_of_classes == 2, config.number_of_classes
        if(config.use_image_augmentations):
            print("using image augmentations")
            train_dataloader = NorwayTransferLearningDataLoader(
                config,
                validation=False,
            )
        else:
            print("not using any image augmentations")
            train_dataloader = NorwayTransferLearningDataLoader(
                config,
                validation=False
            )
        validation_dataloader = NorwayTransferLearningDataLoader(
            config,
            validation=True,
        )
    else:
        print("using our dataset")
        if(config.use_image_augmentations):
            print("using image augmentations")
            train_dataloader = TransferLearningDataLoader(
                config,
                validation=False,

            )
        else:
            print("not using any imgae augmentations")
            train_dataloader = TransferLearningDataLoader(
                config,
                validation=False,
            )
            
        validation_dataloader = TransferLearningDataLoader(
            config,
            validation=True,
        )

    #print the model summary and save it into the output folder
    # model.summary()
    # model_architecture_path = os.path.join(config.summary_dir, "model_architecture")
    # with open(model_architecture_path, "w") as fh:
    #     # Pass the file handle in as a lambda function to make it callable
    #     model.summary(print_fn=lambda x: fh.write(x + "\n"))
    # experiment.log_asset(model_architecture_path)
    experiment.log_asset(args.config)

    save_input_label_and_prediction(model, validation_dataloader, experiment, config, 0, 0)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=config.lr_decay_staircase)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Define metrics and losses
    ###########################
    precision = tf_keras_metrics.Precision() # positive predictive value in the paper
    recall = tf_keras_metrics.Recall() # equivalent to sensitivity in the norway paper

    if(config.number_of_classes == 2):
        print("Doing binary classification")
        loss = tf_keras_losses.binary_crossentropy
        accuracy = tf_keras_metrics.binary_accuracy
        mean_iou_with_argmax = MeanIouWithArgmax(num_classes=2)
        f1_score = F1Score(num_classes=2, average='micro', threshold=0.5)  # dice similarity is equivalent to f1 score
        matthews_corelation_coefficient = MatthewsCorrelationCoefficient()

    elif(config.number_of_classes > 2):
        print("Doing classification with {} classes".format(config.number_of_classes))
        loss = tf_keras_losses.categorical_crossentropy
        accuracy = tf_keras_metrics.categorical_accuracy
        mean_iou_with_argmax = MeanIouWithArgmax(num_classes=config.number_of_classes)
        f1_score = F1Score(num_classes=config.number_of_classes, average='micro')  # dice similarity is equivalent to f1 score
        matthews_corelation_coefficient = MatthewsCorrelationCoefficient()

    else:
        print("Running model for {} classes not supported".format(config.number_of_classes))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[precision, recall, f1_score, matthews_corelation_coefficient, accuracy, mean_iou_with_argmax]
    )

    model.fit(train_dataloader.dataset, epochs=config.num_epochs, steps_per_epoch=len(train_dataloader),
              validation_data=validation_dataloader.dataset,
              validation_steps=len(validation_dataloader),
              callbacks=[EvaluateDuringTraningCallback(validate_every_n_steps=config.validate_every_n_steps,
                                                       validation_dataloader=validation_dataloader,
                                                       comet_experiment=experiment, config=config)],
              use_multiprocessing=False)


def save_input_label_and_prediction(model, validation_dataloader, comet_experiment, config, epoch, step):
    for i, data in enumerate(validation_dataloader.dataset):
        assert data[0][0].numpy().shape == (config.image_size, config.image_size, 3), data[0][0].numpy().shape
        assert data[1][0].numpy().shape == (config.image_size, config.image_size, config.number_of_classes), data[1][0].numpy().shape
        input = data[0][:1] #keep batch dimensions
        label = data[1][0]
        np.save(os.path.join(config.summary_dir, "image.npy"), input.numpy())
        np.save(os.path.join(config.summary_dir, "label.npy"), label.numpy())

        input_np = input[0].numpy().astype('uint8')
        assert input_np.shape == (config.image_size, config.image_size, 3), input_np.shape
        input_image = Image.fromarray(input_np, 'RGB')

        prediction = model.predict(input)
        prediction_np = np.argmax(prediction[0], axis=-1).astype('uint8')
        assert prediction_np.shape == (config.image_size, config.image_size), prediction_np.shape
        prediction_image = Image.fromarray(prediction_np, "P")

        np_label = np.argmax(label.numpy(), -1).astype('uint8')
        assert np_label.shape == (config.image_size, config.image_size), np_label.shape
        label_image = Image.fromarray(np_label, 'P')


        if(config.number_of_classes == 3):
            input_image.putpalette([
                255, 255, 255,  # white
                255, 0, 0,  # red
                0, 0, 255  # blue
            ])
            prediction_image.putpalette([
                255, 255, 255,  # white
                255, 0, 0,  # red
                0, 0, 255  # blue
            ])
            label_image.putpalette([
                255, 255, 255,  # white
                255, 0, 0,  # red
                0, 0, 255  # blue
            ])
        elif(config.number_of_classes ==2):
            prediction_image.putpalette([
                255, 255, 255,  # white
                0, 0, 0,  # black
            ])
            label_image.putpalette([
                255, 255, 255,  # white
                0, 0, 0,  # black
            ])

        else:
            raise NotImplementedError()
        comet_experiment.log_image(input_image, name="input_image_epoch{}_step{}_nr{}".format(epoch, step, i))
        comet_experiment.log_image(prediction_image, name="prediction_image_epoch{}_step{}_nr{}".format(epoch, step, i))
        comet_experiment.log_image(label_image, name="label_image_epoch{}_step{}_nr{}".format(epoch, step, i))

        #image.save(os.path.join(config.summary_dir, "image.png"))
        #label.save(os.path.join(config.summary_dir, "label.png"))
        if(i == 3):
            break

class EvaluateDuringTraningCallback(tf.keras.callbacks.Callback):
    """Callback that evaluates on validation set during training i.e. not only at end of epoch."""

    def __init__(self, validate_every_n_steps, validation_dataloader, comet_experiment, config):
        super(EvaluateDuringTraningCallback, self).__init__()
        self.validate_every_n_steps = validate_every_n_steps
        self.validation_dataloader = validation_dataloader
        self.comet_experiment = comet_experiment
        self.config = config

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.current_epoch = 1

    def on_epoch_end(self, epoch, logs=None):
        self.seen = 0
        self.current_epoch +=1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        # In case of distribution strategy we can potentially run multiple steps
        # at the same time, we should account for that in the `seen` calculation.
        num_steps = logs.get("num_steps", 1)
        self.seen += num_steps

        if (self.seen % self.validate_every_n_steps == 0 and self.seen != 0):
            print("Evaluating on validation set")
            evaluation_metrics = self.model.evaluate(self.validation_dataloader.dataset, steps=len(self.validation_dataloader))
            print("Evaluation metrics", evaluation_metrics)
            for i in range(1, len(evaluation_metrics)):
                self.comet_experiment.log_metric("callback_validation" + self.model.metrics[i - 1].name, evaluation_metrics[i])

            save_input_label_and_prediction(self.model, self.validation_dataloader, self.comet_experiment, self.config,
                                            self.current_epoch, self.seen)


def get_training_augmentations(p=0.6):
    return A.Compose([

        A.RandomRotate90(p=0.5),
        #A.HorizontalFlip(p=0.5),j
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),
        A.IAAEmboss(p=0.05),
        A.Blur(p=0.01, blur_limit=3),
        A.HueSaturationValue(p=1),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),


        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ])
        ], p=p),

if __name__ == "__main__":
    main()
