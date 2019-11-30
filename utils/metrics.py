import tensorflow as tf
import tf.keras.backend as K
from tensorflow.keras.metrics import Metric

class PositivePredictiveValue(Metric):

  def __init__(self, name='positive_predictive_value', **kwargs):
    super(PositivePredictiveValue, self).__init__(name=name, **kwargs)
    self.result = self.add_weight(name='result', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.reshape(y_true, [-1])
      y_pred = tf.reshape(y_pred, [-1])

      tp = tf.compat.v1.metrics.true_positives(y_true, y_pred)
      fp = tf.compat.v1.metrics.false_positives(y_true, y_pred)

      self.result = tp / (tp + fp)

  def result(self):
    return self.result


class Sensitivity(Metric):
  def __init__(self, name='sensitivity', **kwargs):
    super(Sensitivity, self).__init__(name=name, **kwargs)
    self.result = self.add_weight(name='result', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.reshape(y_true, [-1])
      y_pred = tf.reshape(y_pred, [-1])

      tp = tf.compat.v1.metrics.true_positives(y_true, y_pred)
      fn = tf.compat.v1.metrics.false_negatives(y_true, y_pred)

      self.result = tp / (tp + fn)


  def result(self):
    return self.result



class DiceSimilarityCoefcient(Metric):
  def __init__(self, name='dice_similarity_coefcient', **kwargs):
    super(DiceSimilarityCoefcient, self).__init__(name=name, **kwargs)
    self.result = self.add_weight(name='result', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.reshape(y_true, [-1])
      y_pred = tf.reshape(y_pred, [-1])

      tp = tf.compat.v1.metrics.true_positives(y_true, y_pred)
      fp = tf.compat.v1.metrics.false_positives(y_true, y_pred)
      fn = tf.compat.v1.metrics.false_negatives(y_true, y_pred)

      ppv = tp / (tp + fp)
      sen = tp / (tp + fn)

      self.result = (ppv * sen) / (ppv + sen)


  def result(self):
    return self.result

class MatthewsCorrelationCoefcient(Metric):
  def __init__(self, name='matthews_correlation_coefcient', **kwargs):
    super(MatthewsCorrelationCoefcient, self).__init__(name=name, **kwargs)
    self.result = self.add_weight(name='result', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.reshape(y_true, [-1])
      y_pred = tf.reshape(y_pred, [-1])

      tp = tf.compat.v1.metrics.true_positives(y_true, y_pred)
      tn = tf.compat.v1.metrics.true_negatives(y_true, y_pred)
      fp = tf.compat.v1.metrics.false_positives(y_true, y_pred)
      fn = tf.compat.v1.metrics.false_negatives(y_true, y_pred)

      top = (tp * tn) - (fp * fn)
      bottom = tf.math.sqrt( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) )

      self.result = top / bottom


  def result(self):
    return self.result