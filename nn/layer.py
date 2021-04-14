from collections import defaultdict
import numpy as np
np.random.seed(12345)
import tensorflow as tf
tf.set_random_seed(12345)
from libraries import modeling

class Layer(object):
    _name_dict = defaultdict(int)

    def __init__(self, name=None):
        if name is None:
            name = "layer"

        self.name = name + "_" + str(self._name_dict[name] + 1)
        self._name_dict[name] += 1


class Embedding(Layer):
    def __init__(self, pretrained_embedding=None, embedding_shape=None, trainable=True, init_scale=0.02,
                 name="embedding"):
        super(Embedding, self).__init__(name)
        if pretrained_embedding is None and embedding_shape is None:
            raise ValueError("At least one of pretrained_embedding and embedding_shape must be specified!")
        input_shape = pretrained_embedding.shape if pretrained_embedding is not None else embedding_shape

        with tf.variable_scope(self.name):
            embedding_init = tf.constant_initializer(pretrained_embedding) \
                if pretrained_embedding is not None else tf.random_uniform_initializer(-init_scale, init_scale)
            self.embedding = tf.get_variable('embedding', shape=input_shape,
                                             initializer=embedding_init, trainable=trainable)

    def __call__(self, indices):
        return tf.nn.embedding_lookup(self.embedding, indices)


def dropout(x, keep_prob, training, noise_shape=None):
    if keep_prob >= 1.0:
        return x
    return tf.cond(training, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape), lambda: x)


class Dropout(Layer):
    def __init__(self, keep_prob=1.0, name="dropout"):
        super(Dropout, self).__init__(name)
        self.keep_prob = keep_prob

    def __call__(self, x, training):
        return dropout(x, self.keep_prob, training)

class VariationalDropout(Layer):
    def __init__(self, keep_prob=1.0, name="variational_dropout"):
        super(VariationalDropout, self).__init__(name)
        self.keep_prob = keep_prob

    def __call__(self, x, training):
        input_shape = tf.shape(x)
        return dropout(x, self.keep_prob, training, noise_shape=[input_shape[0], 1, input_shape[2]])




def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth)

    return output, attention_weights


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output, attention_weights


class UniAttention(Layer):
    """ Commonly used Uni-Directional Attention"""

    def __init__(self, similarity_function, name="uni_attention"):
        super(UniAttention, self).__init__(name)
        self.similarity_function = similarity_function

    def __call__(self, query, key, key_len, value=None):
        # If value is not given, key will be treated as value
        sim_mat = self.similarity_function(query, key)
        mask = tf.expand_dims(tf.sequence_mask(key_len, tf.shape(key)[1], dtype=tf.float32), axis=1)
        sim_mat = sim_mat + (1. - mask) * tf.float32.min

        sim_prob = tf.nn.softmax(sim_mat)
        if value is not None:
            return tf.matmul(sim_prob, value)
        else:
            return tf.matmul(sim_prob, key)

class ProjectedDotProduct(Layer):
    def __init__(self, hidden_units, activation=None, reuse_weight=False, name="projected_dot_product"):
        super(ProjectedDotProduct, self).__init__(name)
        self.reuse_weight = reuse_weight
        self.projecting_layer = tf.keras.layers.Dense(hidden_units, activation=activation,
                                                      use_bias=False)
        if not reuse_weight:
            self.projecting_layer2 = tf.keras.layers.Dense(hidden_units, activation=activation,
                                                           use_bias=False)

    def __call__(self, t0, t1):
        t0 = self.projecting_layer(t0)
        if self.reuse_weight:
            t1 = self.projecting_layer(t1)
        else:
            t1 = self.projecting_layer2(t1)

        return tf.matmul(t0, t1, transpose_b=True)

import os
class BertEmbedding(Layer):
    def __init__(self, BERT_PRETRAINED_DIR='/uncased_L-12_H-768_A-12/', name='bert_model_helper'):
        super(BertEmbedding, self).__init__(name)
        CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'albert_config_base.json')
        self.bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
        self.init_checkpoint = os.path.join(BERT_PRETRAINED_DIR, 'albert_model.ckpt')

    def __call__(self, input_ids, input_mask, segment_ids, is_training,query_type_ids=None,use_one_hot_embeddings=True,return_pool_output=False,use_fp16=False):
        """Creates a classification model."""
        self.model = modeling.BertModel(
                config=self.bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                query_type_ids = query_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)
        return self.model.get_sequence_output() if not return_pool_output else  (self.model.get_sequence_output(),self.model.get_pooled_output())

    def init_bert(self):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        init_checkpoint = self.init_checkpoint
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                           init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

