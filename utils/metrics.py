import tensorflow as tf

class MeanIouWithArgmax(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        num_classes = tf.shape(y_true)[-1]

        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), num_classes)
        y_true = tf.one_hot(tf.argmax(y_true, axis=-1), num_classes)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name="matthews_correlation_coefcient", **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = tf.shape(y_true)[-1]
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), num_classes)
        y_true = tf.one_hot(tf.argmax(y_true, axis=-1), num_classes)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        self.tp.reset_states()
        self.tp.update_state(y_true, y_pred)

        self.tn.reset_states()
        self.tn.update_state(y_true, y_pred)

        self.fp.reset_states()
        self.fp.update_state(y_true, y_pred)

        self.fn.reset_states()
        self.fn.update_state(y_true, y_pred)

    def result(self):
        tp = self.tp.result()
        tn = self.tn.result()
        fp = self.fp.result()
        fn = self.fn.result()
        top = (tp * tn) - (fp * fn)
        bottom = tf.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return tf.math.divide_no_nan(top, bottom)

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()
