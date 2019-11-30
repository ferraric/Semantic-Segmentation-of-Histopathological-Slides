import tensorflow as tf
from tensorflow.keras.metrics import Metric

class PositivePredictiveValue(Metric):

  def __init__(self, name='positive_predictive_value', **kwargs):
    super(PositivePredictiveValue, self).__init__(name=name, **kwargs)
    self.result = self.add_weight(name='result', initializer='zeros')
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


class Sensitivity(Metric):
  def __init__(self, name='sensitivity', **kwargs):
    super(Sensitivity, self).__init__(name=name, **kwargs)
    self.result = self.add_weight(name='result', initializer='zeros')
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



class DiceSimilarityCoefcient(Metric):
  def __init__(self, name='dice_similarity_coefcient', **kwargs):
    super(DiceSimilarityCoefcient, self).__init__(name=name, **kwargs)
    self.result = self.add_weight(name='result', initializer='zeros')
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

      self.fn.reset_states()
      self.fn.update_state(y_true, y_pred)
      fn = self.fn.result()

      ppv = tp / (tp + fp)
      sen = tp / (tp + fn)

      self.result.assign_add((ppv * sen) / (ppv + sen))


  def result(self):
    return self.result

class MatthewsCorrelationCoefcient(Metric):
  def __init__(self, name='matthews_correlation_coefcient', **kwargs):
    super(MatthewsCorrelationCoefcient, self).__init__(name=name, **kwargs)
    self.result = self.add_weight(name='result', initializer='zeros')
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

      self.tn.reset_states()
      self.tn.update_state(y_true, y_pred)
      tn = self.tn.result()

      self.fp.reset_states()
      self.fp.update_state(y_true, y_pred)
      fp = self.fp.result()

      self.fn.reset_states()
      self.fn.update_state(y_true, y_pred)
      fn = self.fn.result()

      top = (tp * tn) - (fp * fn)
      bottom = tf.math.sqrt( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) )

      self.result.assign_add(top / bottom)


  def result(self):
    return self.result