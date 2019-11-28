import tensorflow as tf
SM_FRAMEWORK="tf.keras"
import segmentation_models as sm
sm.set_framework("tf.keras")


class TransferLearningUnetModel():
    def __init__(self, config):
        self.config = config
        if (self.config.image_net_weights):
            self.encoder_weights = "imagenet"
        else:
            self.encoder_weights = None

        if(self.config.number_of_classes == 2):
            self.activation = 'sigmoid'
        else:
            self.activation = 'softmax'
        self.build_model()


    def build_model(self):
        self.model = sm.Unet(backbone_name=self.config.backbone,
                             input_shape=(self.config.image_size, self.config.image_size, 3),
                             classes=self.config.number_of_classes,
                             activation=self.activation,
                             encoder_weights=self.encoder_weights,
                             decoder_block_type=self.config.decoder_block_type)
