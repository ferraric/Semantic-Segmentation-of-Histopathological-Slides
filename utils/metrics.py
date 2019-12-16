import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric


class PositivePredictiveValue(Metric):
    def __init__(self, num_classes, name='positive_predictive_value', **kwargs):
        super(PositivePredictiveValue, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.value = self.add_weight(name='value', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_true = tf.dtypes.cast(tf.reshape(y_true, [-1]), dtype=tf.float64)
        y_pred = tf.dtypes.cast(tf.reshape(y_pred, [-1]), dtype=tf.float64)

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)

        tp = tf.linalg.diag_part(cm)
        fp = tf.math.reduce_sum(cm, axis=1) - tp

        tp = tf.math.reduce_sum(tp)
        fp = tf.math.reduce_sum(fp)

        self.value.assign_add(-self.value)
        self.value.assign_add(tp / (tp + fp))

    def result(self):
        return tf.math.reduce_mean(self.value)



class Sensitivity(Metric):
    def __init__(self, num_classes, name='sensitivity', **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.value = self.add_weight(name='value', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_true = tf.dtypes.cast(tf.reshape(y_true, [-1]), dtype=tf.float64)
        y_pred = tf.dtypes.cast(tf.reshape(y_pred, [-1]), dtype=tf.float64)

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)

        tp = tf.linalg.diag_part(cm)
        fn = tf.math.reduce_sum(cm, axis=0) - tp     

        tp = tf.math.reduce_sum(tp)
        fn = tf.math.reduce_sum(fn)

        self.value.assign_add(-self.value)
        self.value.assign_add(tp / (tp + fn))

    def result(self):
        return tf.math.reduce_mean(self.value)




class DiceSimilarityCoefcient(Metric):
    def __init__(self, num_classes, name='dice_similarity_coefcient', **kwargs):
        super(DiceSimilarityCoefcient, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.value = self.add_weight(name='value', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_true = tf.dtypes.cast(tf.reshape(y_true, [-1]), dtype=tf.float64)
        y_pred = tf.dtypes.cast(tf.reshape(y_pred, [-1]), dtype=tf.float64)

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)

        tp = tf.linalg.diag_part(cm)
        fp = tf.math.reduce_sum(cm, axis=1) - tp
        fn = tf.math.reduce_sum(cm, axis=0) - tp     

        tp = tf.math.reduce_sum(tp)
        fp = tf.math.reduce_sum(fp)
        fn = tf.math.reduce_sum(fn)

        ppv = tp / (tp + fp)
        sen = tp / (tp + fn)

        self.value.assign_add(-self.value)
        self.value.assign_add(2 * (ppv * sen) / (ppv + sen))

    def result(self):
        return tf.math.reduce_mean(self.value)


class MatthewsCorrelationCoefcient(Metric):
    def __init__(self, num_classes, name='matthews_correlation_coefcient', **kwargs):
        super(MatthewsCorrelationCoefcient, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.value = self.add_weight(name='value', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_true = tf.dtypes.cast(tf.reshape(y_true, [-1]), dtype=tf.float64)
        y_pred = tf.dtypes.cast(tf.reshape(y_pred, [-1]), dtype=tf.float64)

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)

        tp = tf.linalg.diag_part(cm)
        fp = tf.math.reduce_sum(cm, axis=1) - tp
        fn = tf.math.reduce_sum(cm, axis=0) - tp     
        tn = tf.tile(tf.reshape(tf.math.reduce_sum(cm), [1]), tf.shape(tp)) - tp - fp - fn

        tp = tf.math.reduce_sum(tp)
        fp = tf.math.reduce_sum(fp)
        fn = tf.math.reduce_sum(fn)
        tn = tf.math.reduce_sum(tn)


        top = tf.dtypes.cast( (tp * tn) - (fp * fn), dtype=tf.float64)
        bottom = tf.math.sqrt( tf.dtypes.cast( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn), dtype=tf.float64) )

        self.value.assign_add(-self.value)
        self.value.assign_add(tf.math.divide_no_nan(top, bottom))      

    def result(self):
        return tf.math.reduce_mean(self.value)