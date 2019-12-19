#coding:utf-8
import tensorflow as tf
def focal_loss_softmax(labels, logits, gamma=2,alpha=0.5):
    y_pred = tf.nn.softmax(logits)
    labels = tf.one_hot(labels, depth=tf.shape(y_pred)[1])

    loss = -alpha*labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    return tf.reduce_sum(loss, axis=1)

def get_l2_loss():
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                       if 'bias' not in v.name]) * 0.001
    return lossL2

def _smooth_one_hot_labels(logits, labels, label_smoothing):
  label_smoothing = tf.constant(label_smoothing, dtype=logits.dtype)
  num_classes = tf.shape(logits)[-1]
  return tf.one_hot(
      tf.cast(labels, tf.int32),
      num_classes,
      on_value=1.0 - label_smoothing,
      off_value=label_smoothing / tf.cast(num_classes - 1, label_smoothing.dtype),
      dtype=logits.dtype)

def softmax_with_logits_label_smooth(logits,labels,label_smoothing=0.1):
    smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
    smoothed_labels = tf.stop_gradient(smoothed_labels)
    return  tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=smoothed_labels)
        )

