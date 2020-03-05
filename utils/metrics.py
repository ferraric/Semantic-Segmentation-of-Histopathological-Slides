from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
import numpy as np


class MeanIouWithArgmax(tf.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        num_classes = tf.shape(y_true)[-1]

        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), num_classes)
        y_true = tf.one_hot(tf.argmax(y_true, axis=-1), num_classes)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


# result computation adapted from https://github.com/vlainic/matthews-correlation-coefficient
class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="matthews_correlation_coefcient", **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.reset_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        new_confusion_matrix = tf.math.confusion_matrix(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]),
                                                        num_classes=self.num_classes, dtype=tf.dtypes.float32)
        assert self.confusion_matrix.shape == new_confusion_matrix.shape
        self.confusion_matrix.assign(self.confusion_matrix + new_confusion_matrix)

    def result(self):
        N = K.sum(self.confusion_matrix)
        up = N * tf.linalg.trace(self.confusion_matrix) - K.sum(tf.matmul(self.confusion_matrix, self.confusion_matrix))
        down_left = K.sqrt(N ** 2 - K.sum(tf.matmul(self.confusion_matrix, K.transpose(self.confusion_matrix))))
        down_right = K.sqrt(N ** 2 - K.sum(tf.matmul(K.transpose(self.confusion_matrix), self.confusion_matrix)))

        mcc = up / (down_left * down_right + K.epsilon())
        mcc = tf.where(tf.math.is_nan(mcc), tf.zeros_like(mcc), mcc)

        return K.mean(mcc)

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.confusion_matrix = tf.Variable(tf.zeros((self.num_classes, self.num_classes)), name="confusion_matrix")


class FBetaScore(tf.keras.metrics.Metric):
    """Computes F-Beta score.
    It is the weighted harmonic mean of precision
    and recall. Output range is [0, 1]. Works for
    both multi-class and multi-label classification.
    F-Beta = (1 + beta^2) * (prec * recall) / ((beta^2 * prec) + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-Beta Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(self,
                 num_classes,
                 average=None,
                 beta=1.0,
                 threshold=None,
                 name='fbeta_score',
                 dtype=tf.float32):
        super(FBetaScore, self).__init__(name=name)

        if average not in (None, 'micro', 'macro', 'weighted'):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [None, micro, macro, weighted]")

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError(
                    "The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != 'micro':
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name,
                shape=self.init_shape,
                initializer='zeros',
                dtype=self.dtype)

        self.true_positives = _zero_wt_init('true_positives')
        self.false_positives = _zero_wt_init('false_positives')
        self.false_negatives = _zero_wt_init('false_negatives')
        self.weights_intermediate = _zero_wt_init('weights_intermediate')

    # TODO: Add sample_weight support, currently it is
    # ignored during calculations.
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold,
                                    tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        def _count_non_zero(val):
            non_zeros = tf.math.count_nonzero(val, axis=self.axis)
            return tf.cast(non_zeros, self.dtype)

        self.true_positives.assign_add(_count_non_zero(y_pred * y_true))
        self.false_positives.assign_add(_count_non_zero(y_pred * (y_true - 1)))
        self.false_negatives.assign_add(_count_non_zero((y_pred - 1) * y_true))
        self.weights_intermediate.assign_add(_count_non_zero(y_true))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives)
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = (tf.math.divide_no_nan(mul_value, add_value))
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == 'weighted':
            weights = tf.math.divide_no_nan(
                self.weights_intermediate,
                tf.reduce_sum(self.weights_intermediate))
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
        }

        if self.threshold is not None:
            config["threshold"] = self.threshold

        base_config = super(FBetaScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
        self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))


class F1Score(FBetaScore):
    """Computes F-1 Score.
    It is the harmonic mean of precision and recall.
    Output range is [0, 1]. Works for both multi-class
    and multi-label classification.
    F-1 = 2 * (precision * recall) / (precision + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro`
            and `weighted`. Default value is None.
        threshold: Elements of `y_pred` above threshold are
            considered to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-1 Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(self,
                 num_classes,
                 average=None,
                 threshold=None,
                 name='f1_score',
                 dtype=tf.float32):
        super(F1Score, self).__init__(
            num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def get_config(self):
        base_config = super(F1Score, self).get_config()
        del base_config["beta"]
        return base_config
