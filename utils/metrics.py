import tensorflow as tf

class PositivePredictiveValue(tf.keras.metrics.Metric):
    def __init__(self, name="positive_predictive_value", **kwargs):
        super(PositivePredictiveValue, self).__init__(name=name, **kwargs)
        self.result = self.add_weight(name="result", initializer="zeros")
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        self.tp.reset_states()
        self.tp.update_state(y_true, y_pred)
        tp = self.tp.result()

        self.fp.reset_states()
        self.fp.update_state(y_true, y_pred)
        fp = self.fp.result()

        self.result.assign_add(tp / (tp + fp))

    def result(self):
        return self.result


class Sensitivity(tf.keras.metrics.Metric):
    def __init__(self, name="sensitivity", **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.result = self.add_weight(name="result", initializer="zeros")
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        self.tp.reset_states()
        self.tp.update_state(y_true, y_pred)
        tp = self.tp.result()

        self.fn.reset_states()
        self.fn.update_state(y_true, y_pred)
        fn = self.fn.result()

        self.result.assign_add(tp / (tp + fn))

    def result(self):
        return self.result


class DiceSimilarityCoefcient(tf.keras.metrics.Metric):
    def __init__(self, name="dice_similarity_coefcient", **kwargs):
        super(DiceSimilarityCoefcient, self).__init__(name=name, **kwargs)
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        self.tp.reset_states()
        self.tp.update_state(y_true, y_pred)

        self.fp.reset_states()
        self.fp.update_state(y_true, y_pred)

        self.fn.reset_states()
        self.fn.update_state(y_true, y_pred)

    def result(self):
        tp = self.tp.result()
        fp = self.fp.result()
        fn = self.fn.result()

        ppv = tp / (tp + fp)
        sen = tp / (tp + fn)
        result = 2 * (ppv * sen) / (ppv + sen)
        return result

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()



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
        return top / bottom

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.tp = tf.keras.metrics.TruePositives()
        self.tn = tf.keras.metrics.TrueNegatives()
        self.fp = tf.keras.metrics.FalsePositives()
        self.fn = tf.keras.metrics.FalseNegatives()


#
class MeanIouWithArgmax(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        num_classes = tf.shape(y_true)[-1]

        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), num_classes)
        y_true = tf.one_hot(tf.argmax(y_true, axis=-1), num_classes)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)
