import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, add


class StackedUNetModel(Model):
    """This is a simple example class that shows how to use the project architecture for semantic segmentation. """

    def __init__(self, config):
        super(StackedUNetModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        '''Build U-Net model'''
        self.inputlayer = InputLayer(
            input_shape=(self.config.patch_size, self.config.patch_size, self.config.patch_channels))

        # first Unet
        self.conv11_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv12_1 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.pool1_1 = MaxPooling2D((2, 2), strides=2)

        self.conv21_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv22_1 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.pool2_1 = MaxPooling2D((2, 2), strides=2)

        self.conv31_1 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv32_1 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.pool3_1 = MaxPooling2D((2, 2), strides=2)

        self.conv41_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv42_1 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.drop4_1 = Dropout(0.5)  # Not in Paper, but is in code
        self.pool4_1 = MaxPooling2D((2, 2), strides=2)

        self.conv51_1 = Conv2D(1024, (3, 3), activation='relu', padding='same')
        self.conv52_1 = Conv2D(1024, (3, 3), activation='relu', padding='same')
        self.drop5_1 = Dropout(0.5)  # Not in Paper, but is in code
        self.upconv5_1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')

        # self.concatenate6 = Concatenate([self.upconv5, self.drop4])
        self.conv61_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv62_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.upconv6_1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')

        # self.concatenate7 = Concatenate([self.upconv6, self.conv32])
        self.conv71_1 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv72_1 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.upconv7_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')

        # self.concatenate8 = Concatenate([self.upconv7, self.conv22])
        self.conv81_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv82_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.upconv8_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')

        # self.concatenate9 = Concatenate([self.upconv8, self.conv12])
        self.conv91_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv92_1 = Conv2D(64, (3, 3), activation='relu', padding='same')

        self.conv93_1 = Conv2D(self.config.number_of_classes, (1, 1), activation='softmax', padding='same')

        # second Unet
        self.conv11_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv12_2 = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.pool1_2 = MaxPooling2D((2, 2), strides=2)

        self.conv21_2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv22_2 = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.pool2_2 = MaxPooling2D((2, 2), strides=2)

        self.conv31_2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv32_2 = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.pool3_2 = MaxPooling2D((2, 2), strides=2)

        self.conv41_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv42_2 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2)
        self.drop4_2 = Dropout(0.5)  # Not in Paper, but is in code
        self.pool4_2 = MaxPooling2D((2, 2), strides=2)

        self.conv51_2 = Conv2D(1024, (3, 3), activation='relu', padding='same')
        self.conv52_2 = Conv2D(1024, (3, 3), activation='relu', padding='same')
        self.drop5_2 = Dropout(0.5)  # Not in Paper, but is in code
        self.upconv5_2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')

        # self.concatenate6 = Concatenate([self.upconv5, self.drop4])
        self.conv61_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv62_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.upconv6_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')

        # self.concatenate7 = Concatenate([self.upconv6, self.conv32])
        self.conv71_2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv72_2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.upconv7_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')

        # self.concatenate8 = Concatenate([self.upconv7, self.conv22])
        self.conv81_2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv82_2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.upconv8_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')

        # self.concatenate9 = Concatenate([self.upconv8, self.conv12])
        self.conv91_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv92_2 = Conv2D(64, (3, 3), activation='relu', padding='same')

        self.conv93_2 = Conv2D(self.config.number_of_classes, (1, 1), activation='softmax', padding='same')

        self.out_comb = Conv2D(self.config.number_of_classes, (1, 1), activation='softmax', padding='same')

    def call(self, x):
        x = self.inputlayer(x)
        x = self.conv11_1(x)
        x = self.conv12_1(x)
        conv12_1 = x
        x = self.pool1_1(x)

        x = self.conv21_1(x)
        x = self.conv22_1(x)
        conv22_1 = x
        x = self.pool2_1(x)

        x = self.conv31_1(x)
        x = self.conv32_1(x)
        conv32_1 = x
        x = self.pool3_1(x)

        x = self.conv41_1(x)
        x = self.conv42_1(x)
        x = self.drop4_1(x)
        drop4_1 = x
        x = self.pool4_1(x)

        x = self.conv51_1(x)
        x = self.conv52_1(x)
        x = self.drop5_1(x)
        x = self.upconv5_1(x)

        x = concatenate([x, drop4_1])
        x = self.conv61_1(x)
        x = self.conv62_1(x)
        x = self.upconv6_1(x)

        x = concatenate([x, conv32_1])
        x = self.conv71_1(x)
        x = self.conv72_1(x)
        x = self.upconv7_1(x)

        x = concatenate([x, conv22_1])
        x = self.conv81_1(x)
        x = self.conv82_1(x)
        x = self.upconv8_1(x)
        x = concatenate([x, conv12_1])
        x = self.conv91_1(x)
        x = self.conv92_1(x)

        x = self.conv93_1(x)
        intermediate_output = x

        x = self.conv11_2(x)
        x = self.conv12_2(x)
        conv12_2 = x
        x = self.pool1_2(x)

        x = self.conv21_2(x)
        x = self.conv22_2(x)
        conv22_2 = x
        x = self.pool2_2(x)

        x = self.conv31_2(x)
        x = self.conv32_2(x)
        conv32_2 = x
        x = self.pool3_2(x)

        x = self.conv41_2(x)
        x = self.conv42_2(x)
        x = self.drop4_2(x)
        drop4_2 = x
        x = self.pool4_2(x)

        x = self.conv51_2(x)
        x = self.conv52_2(x)
        x = self.drop5_2(x)
        x = self.upconv5_2(x)

        x = concatenate([x, drop4_2])
        x = self.conv61_2(x)
        x = self.conv62_2(x)
        x = self.upconv6_2(x)

        x = concatenate([x, conv32_2])
        x = self.conv71_2(x)
        x = self.conv72_2(x)
        x = self.upconv7_2(x)

        x = concatenate([x, conv22_2])
        x = self.conv81_2(x)
        x = self.conv82_2(x)
        x = self.upconv8_2(x)
        x = concatenate([x, conv12_2])
        x = self.conv91_2(x)
        x = self.conv92_2(x)

        x = self.conv93_2(x)

        x = concatenate([x, intermediate_output])
        x = self.out_comb(x)
        return x
