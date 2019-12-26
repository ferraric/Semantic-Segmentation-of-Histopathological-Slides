from tensorflow.keras.layers import Layer, Add, BatchNormalization


class BNBlock(Layer):
    def __init__(self):
        super(BNBlock, self).__init__(name='')

    def build(self, input_shape):
        self.add = Add()
        self.bn = BatchNormalization(axis=-1)

    def call(self, x):
        x = self.add([x, self.bn(x)])
        return x
