import tensorflow as tf
from nn.layer import Layer


class BaseBiRNN(Layer):
    def __init__(self, name="base_BiRNN"):
        super(BaseBiRNN, self).__init__(name)
        self.fw_cell = None
        self.bw_cell = None

    def __call__(self, seq, seq_len):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, seq, seq_len, dtype=tf.float32)
        if isinstance(states[0], tuple):
            return tf.concat(outputs, axis=-1), tf.concat([s.h for s in states], axis=-1)
        else:
            return tf.concat(outputs, axis=-1), tf.concat(states, axis=-1)


class BiLSTM(BaseBiRNN):
    def __init__(self, hidden_units, name="BiLSTM"):
        super(BiLSTM, self).__init__(name)
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, name=name + '_fw_cell')
        self.bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, name=name + '_bw_cell')


class BiGRU(BaseBiRNN):
    def __init__(self, hidden_units, name='BiGRU'):
        super(BiGRU, self).__init__(name)
        self.fw_cell = tf.nn.rnn_cell.GRUCell(hidden_units, name=name + '_fw_cell')
        self.bw_cell = tf.nn.rnn_cell.GRUCell(hidden_units, name=name + '_bw_cell')
