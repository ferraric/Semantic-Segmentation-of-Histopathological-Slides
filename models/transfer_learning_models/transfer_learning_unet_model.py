import tensorflow as tf
SM_FRAMEWORK= tf.keras
import models.transfer_learning_models.transfer_learning_implementations as sm


class TransferLearningUnetModel():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        self.model = sm.Unet(backbone_name=self.config.backbone, classes=3)
