import models.transfer_learning_models.transfer_learning_implementations as sm
import tensorflow as tf


class TransferLearningUnetModel():
    def __init__(self, config):
        self.config = config
        if (self.config.use_backone_encoder_weights):
            self.encoder_weights = "imagenet"
        else:
            self.encoder_weights = None
        self.build_model()


    def build_model(self):
        self.model = sm.Unet(backbone_name=self.config.backbone, classes=self.config.number_of_classes,
                             encoder_weights=self.encoder_weights)
