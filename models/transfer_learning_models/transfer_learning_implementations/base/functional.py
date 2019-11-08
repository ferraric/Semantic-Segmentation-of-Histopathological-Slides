import tensorflow as tf
SMOOTH = 1e-5

# ----------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------

def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""
    backend = kwargs['backend']
    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (3, 0, 1, 2))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    return x


def get_reduce_axes(per_image, **kwargs):
    backend = kwargs['backend']
    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs


def round_if_needed(x, threshold, **kwargs):
    backend = kwargs['backend']
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x


def average(x, per_image=False, class_weights=None, **kwargs):
    backend = kwargs['backend']
    if per_image:
        x = backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return backend.mean(x)


# ----------------------------------------------------------------
#   Metric Functions
# ----------------------------------------------------------------

def iou_score(gt, pr, class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        IoU/Jaccard score in range [0, 1]

    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index

    """

    backend = kwargs['backend']
    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # score calculation
    intersection = backend.sum(gt * pr, axis=axes)
    union = backend.sum(gt + pr, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None,
            **kwargs):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:

    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        F-score in range [0, 1]

    """

    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # calculate score
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    fn = backend.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def precision(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
    r"""Calculate precision between the ground truth (gt) and the prediction (pr).

    .. math:: F_\beta(tp, fp) = \frac{tp} {(tp + fp)}

    where:
         - tp - true positives;
         - fp - false positives;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.

    Returns:
        float: precision score
    """
    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # score calculation
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    score = (tp + smooth) / (tp + fp + smooth)

    score = average(score, per_image, class_weights, **kwargs)

    return score


def recall(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
    r"""Calculate recall between the ground truth (gt) and the prediction (pr).

    .. math:: F_\beta(tp, fn) = \frac{tp} {(tp + fn)}

    where:
         - tp - true positives;
         - fp - false positives;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.

    Returns:
        float: recall score
    """
    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    tp = backend.sum(gt * pr, axis=axes)
    fn = backend.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score

def matthewscorrelation(gt, pr, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
    r"""Calculate matthewscorrelation between the ground truth (gt) and the prediction (pr).

    .. math:: F_\beta(tp, tn, fp, fn) = \frac{tp \cdot tn - fp  \cdot fn} {\sqrt((tp + fp)(tp + fn)(tn + fp)(tn + fn))}

    where:
         - tp - true positives;
         - fp - false positives;
         - tn - true negatives;
         - fp - false positives;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.

    Returns:
        float: recall score
    """
    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)
    # TODO make sure that this is calculated correctly
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    fn = backend.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights, **kwargs)
    raise NotImplementedError()
    return score


# ----------------------------------------------------------------
#   Loss Functions
# ----------------------------------------------------------------

def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None, **kwargs):
    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # scale predictions so that the class probas of each sample sum to 1
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    pr /= backend.sum(pr, axis=axis, keepdims=True)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1 - backend.epsilon())

    # calculate loss
    output = gt * backend.log(pr) * class_weights
    return - backend.mean(output)


def binary_crossentropy(gt, pr, **kwargs):
    backend = kwargs['backend']
    return backend.mean(backend.binary_crossentropy(gt, pr))


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None, **kwargs):
    r"""Implementation of Focal Loss from the paper in multiclass classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

    """

    backend = kwargs['backend']
    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate focal loss
    loss = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))

    return backend.mean(loss)


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25, **kwargs):
    r"""Implementation of Focal Loss from the paper in binary classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \
               - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0

    """
    backend = kwargs['backend']

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

    loss_1 = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))
    loss_0 = - (1 - gt) * (alpha * backend.pow((pr), gamma) * backend.log(1 - pr))
    loss = backend.mean(loss_0 + loss_1)
    return loss

def mean_iou_with_argmax(y_true, y_pred, **kwargs):
    """
    First calculate the argmax of the prediction.  Then Compute mean Intersection over Union via Keras.
    IoU(A,B) = |A & B| / (| A U B|)
    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    backend = kwargs['backend']
    return calculate_mean_iou_or_dice(y_true, y_pred, metric_name='iou', backend=backend)


def calculate_mean_iou_or_dice(y_true, y_pred, metric_name, metric_type='standard', drop_last=False, backend=None):
    """
    Compute mean metrics of two segmentation masks, via Keras.

    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)

    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)

    Returns:
        IoU/Dice of y_true and y_pred, as a float
          in which case it returns the per-class metric, averaged over the batch.

    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """

    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')

    # always assume one or more classes
    num_classes = backend.shape(y_true)[-1]

    if not flag_soft:
        # get one-hot encoded masks from y_pred (true masks should already be one-hot)
        y_pred = backend.one_hot(backend.argmax(y_pred), num_classes)
        y_true = backend.one_hot(backend.argmax(y_true), num_classes)

    # if already one-hot, could have skipped above command
    # keras uses float32 instead of float64, would give error down (but numpy arrays or keras.to_categorical gives float64)
    y_true = backend.cast(y_true, 'float32')
    y_pred = backend.cast(y_pred, 'float32')

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1, 2)  # W,H axes of each image
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=axes)
    mask_sum = backend.sum(backend.abs(y_true), axis=axes) + backend.sum(backend.abs(y_pred), axis=axes)
    union = mask_sum - intersection  # or, np.logical_or(y_pred, y_true) for one-hot

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    metric = {'iou': iou, 'dice': dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask = backend.cast(backend.not_equal(union, 0), 'float32')

    if drop_last:
        metric = metric[:, :-1]
        mask = mask[:, :-1]


    # return mean metrics: remaining axes are (batch, classes)
    if flag_naive_mean:
        return backend.mean(metric)

    # take mean only over non-absent classes
    class_count = backend.sum(mask, axis=0)
    non_zero = tf.greater(class_count, 0)
    non_zero_sum = tf.boolean_mask(backend.sum(metric * mask, axis=0), non_zero)
    non_zero_count = tf.boolean_mask(class_count, non_zero)

    return backend.mean(non_zero_sum / non_zero_count)
