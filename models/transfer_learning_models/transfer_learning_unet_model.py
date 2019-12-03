import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import tensorflow as tf


class TransferLearningUnetModel():
    def __init__(self, config):
        self.config = config
        if (self.config.use_backone_encoder_weights):
            self.encoder_weights = "imagenet"
        else:
            self.encoder_weights = None

        if(self.config.number_of_classes == 2):
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'

        self.build_model()


    def build_model(self):
        self.model = sm.Unet(backbone_name=self.config.backbone, classes=self.config.number_of_classes,
                             activation=self.activation,
                             encoder_weights=self.encoder_weights)
        # for layer in self.model.layers:
        #     if(isinstance(layer, tf.keras.layers.BatchNormalization )):
        #         print(layer.momentum)
        #         print("setting momentum ")
        #         layer.momentum = 0.1
        #         print(layer.momentum)
