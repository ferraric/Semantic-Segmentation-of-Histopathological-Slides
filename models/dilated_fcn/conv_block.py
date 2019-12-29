from tensorflow.keras.layers import Layer, Conv2D, Activation
from models.dilated_fcn.bn_block import BNBlock


class ConvBlock(Layer):
    def __init__(self, filters, kernel_size, dilation_rate=(1, 1)):
        super(ConvBlock, self).__init__(name='')
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.conv = Conv2D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate,
                           kernel_initializer='he_normal',
                           padding='same')
        self.bn_block = BNBlock()
        self.act = Activation('relu')

    def call(self, x):
        x = self.conv(x)
        x = self.bn_block(x)
        x = self.act(x)
        return x
