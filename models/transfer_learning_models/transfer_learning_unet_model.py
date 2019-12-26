import os
import sys
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import tensorflow as tf


class TransferLearningUnetModel():
    def __init__(self, config):
        self.config = config
        self.model_type = config.model_type
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
        print("We use a {} model with the backbone {}".format(self.model_type, self.config.backbone))
        if(self.model_type == "unet"):
            self.model = sm.Unet(backbone_name=self.config.backbone, input_shape=(None, None,
                                                                                  3),
                                 classes=self.config.number_of_classes,
                                 activation=self.activation,
                                 encoder_weights=self.encoder_weights,
                                 encoder_freeze=False,
                                 decoder_block_type=self.config.decoder_block_type)

        elif(self.model_type == "linknet"):
            self.model = sm.Linknet(backbone_name=self.config.backbone, input_shape=(self.config.image_size, self.config.image_size,
                                                                                  3),
                                 classes=self.config.number_of_classes,
                                 activation=self.activation,
                                 encoder_weights=self.encoder_weights,
                                 encoder_freeze=False,
                                 decoder_block_type=self.config.decoder_block_type)


        elif(self.model_type == "pspnet"):
            assert self.config.pspnet_downsample_factor != None, "Parameter psp_downsample_factor does not exist"
            assert self.config.image_size % (self.config.pspnet_downsample_factor*6) == 0, "Image size should be divisible by donwsamplefactor * 6 but is not {} * {}".format(self.config.pspnet_downsample_factor, 6)
            self.model = sm.PSPNet(backbone_name=self.config.backbone,
                                input_shape=(self.config.image_size, self.config.image_size,
                                             3),
                                classes=self.config.number_of_classes,
                                activation=self.activation,
                                encoder_weights=self.encoder_weights,
                                encoder_freeze=False,
                                downsample_factor=self.config.pspnet_downsample_factor,
                                psp_use_batchnorm=True)


        elif(self.model_type == "fpn"):
            assert self.config.image_size % 32 == 0, "Image size should be divisible by 32 but is not"
            self.model = sm.FPN(backbone_name=self.config.backbone,
                                input_shape=(self.config.image_size, self.config.image_size,
                                             3),
                                classes=self.config.number_of_classes,
                                activation=self.activation,
                                encoder_weights=self.encoder_weights,
                                encoder_freeze=False,
                                pyramid_use_batchnorm=True,
                                pyramid_dropout=0.3)

        else:
            print("The model type {} does not exit".format(self.model_type))
            sys.exit()

        # for layer in self.model.layers:
        #     if(isinstance(layer, tf.keras.layers.BatchNormalization )):
        #         print(layer.momentum)
        #         print("setting momentum ")
        #         layer.momentum = 0.1
        #         print(layer.momentum)
