from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Concatenate, Dropout

from models.dilated_fcn.bn_block import BNBlock
from models.dilated_fcn.conv_block import ConvBlock


class DilatedFcnModel(Model):

    def __init__(self, config):
        super(DilatedFcnModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.bn1 = BNBlock()
        self.conv1 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(1, 1))
        self.conv2 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(1, 1))
        self.conv3 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(2, 2))
        self.conv4 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(3, 3))
        self.conv5 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(5, 5))
        self.conv6 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(8, 8))
        self.conv7 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(13, 13))
        self.conv8 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(21, 21))
        self.conv9 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(34, 34))
        self.conv10 = ConvBlock(filters=32, kernel_size=(3, 3), dilation_rate=(55, 55))
        self.conc = Concatenate(axis=-1)
        self.drop = Dropout(0.5)
        self.conv11 = ConvBlock(filters=128, kernel_size=(1, 1))
        self.conv12 = ConvBlock(filters=32, kernel_size=(1, 1))
        self.conv_out = Conv2D(self.config.number_of_classes, (1, 1), activation='softmax',
                               kernel_initializer='he_normal')

    def call(self, x):
        x = self.bn1(x)
        bn1 = x
        x = self.conv1(x)
        c1 = x
        x = self.conv2(x)
        c2 = x
        x = self.conv3(x)
        c3 = x
        x = self.conv4(x)
        c4 = x
        x = self.conv5(x)
        c5 = x
        x = self.conv6(x)
        c6 = x
        x = self.conv7(x)
        c7 = x
        x = self.conv8(x)
        c8 = x
        x = self.conv9(x)
        c9 = x
        x = self.conv10(x)
        c10 = x
        x = self.conc([bn1, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
        x = self.drop(x)
        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv_out(x)
        return x
