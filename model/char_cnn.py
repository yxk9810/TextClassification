#coding:utf-8
from model.base import BaseModel
import tensorflow as tf
from nn.layer import Embedding,Dropout
from nn.recurrent import BiLSTM
from collections import OrderedDict
from nn.layer import MultiHeadAttention,create_padding_mask
class CharCNN(BaseModel):
    def __init__(self, vocab, pretrained_word_embedding=None,
             word_embedding_size=100,
             rnn_hidden_size = 64,
             dropout_keep_prob=0.9, num_class=3):
        super(CharCNN, self).__init__(vocab)
        self.keep_prob = dropout_keep_prob
        self.num_class = num_class
        self.word_embedding_size = word_embedding_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.word_embedding_trainable = True
        self.rnn_hidden_size =rnn_hidden_size
        self.kernel_size = [3,3]
        self._build_graph()

    def _build_graph(self):
        self.x = tf.placeholder(tf.int32, [None, None])
        self.x_len = tf.placeholder(tf.int32,[None])
        self.y = tf.placeholder(tf.int32, [None])
        self.training = tf.placeholder_with_default(False, shape=(), name='is_training')

        self.filters = 100

        divisors = tf.pow(tf.constant([10000.0] * (self.filters // 2), dtype=tf.float32),
                          tf.range(0, self.filters, 2, dtype=tf.float32) / self.filters)
        quotients = tf.cast(tf.expand_dims(tf.range(0, tf.reduce_max(self.x_len)), -1), tf.float32) / tf.expand_dims(divisors, 0)
        position_repr = tf.concat([tf.sin(quotients), tf.cos(quotients)], -1)


        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(self.vocab.get_word_vocab() + 1, self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)

        input_x = word_embedding(self.x)
        dropout = Dropout(self.keep_prob)
        input_x = dropout(input_x, self.training)
        for layer in range(len(self.kernel_size)):
            norm_x = tf.layers.batch_normalization(input_x)
            input_x+=tf.layers.conv1d(norm_x,self.filters,self.kernel_size[layer],strides=1,padding='SAME',activation=tf.nn.relu)

        input_x = input_x+position_repr
        mask = create_padding_mask(self.x)


        tmp_ma = MultiHeadAttention(self.filters,5)
        norm_x = tf.layers.batch_normalization(input_x)
        mha_out,_ = tmp_ma(norm_x,norm_x,norm_x,mask=mask)
        input_x+=mha_out

        tmp_ma2 = MultiHeadAttention(self.filters,5)
        norm_x = tf.layers.batch_normalization(input_x)
        mha2_out,_ = tmp_ma2(norm_x,norm_x,norm_x,mask=mask)
        input_x+=mha2_out

        # encoder1 = BiLSTM(self.rnn_hidden_size,name='layer_1')
        # input_x,_ = encoder1(input_x,self.x_len)
        #
        # encoder2 = BiLSTM(self.rnn_hidden_size,name='layer_2')
        # input_x,_ = encoder2(input_x,self.x_len)
        # print(input_x.shape)
        # merge = tf.reshape(input_x,[tf.shape(input_x)[0],-1])

        avg_pool = tf.reduce_mean(input_x,axis=1)
        avg_max =  tf.reduce_max(input_x,axis=1)

        merge = tf.concat([avg_pool,avg_max],axis=1)
        # print(merge.shape)
        # h_conc_linear1 = tf.keras.layers.Dense(200,use_bias=False,activation=tf.nn.relu)(merge)
        # h_conc_linear2 = tf.keras.layers.Dense(200,use_bias=False,activation=tf.nn.relu)(merge)
        # merge = merge+h_conc_linear1+h_conc_linear2

        #
        # dense = tf.keras.layers.Dense(16,activation=tf.nn.relu)
        # merge = dense(merge)
        # merge = dropout(merge, self.training)
        self.logits = tf.keras.layers.Dense(self.num_class,activation=None)(merge)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.y))
        global_step = tf.train.get_or_create_global_step()

        self.input_placeholder_dict = OrderedDict({
            "char_ids": self.x,
            "labels": self.y,
            "text_len":self.x_len,
            "training": self.training
        })

        self.output_variable_dict = OrderedDict({
            "predict": tf.argmax(self.logits, axis=1)
        })

        # 8. Metrics and summary
        with tf.variable_scope("train_metrics"):
            self.train_metrics = {
                'loss': tf.metrics.mean(self.loss)
            }

        self.train_update_metrics = tf.group(*[op for _, op in self.train_metrics.values()])
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_metrics")
        self.train_metric_init_op = tf.variables_initializer(metric_variables)

        with tf.variable_scope("eval_metrics"):
            self.eval_metrics = {
                'loss': tf.metrics.mean(self.loss)
            }

        self.eval_update_metrics = tf.group(*[op for _, op in self.eval_metrics.values()])
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="eval_metrics")
        self.eval_metric_init_op = tf.variables_initializer(metric_variables)

        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

    def compile(self, optimizer, initial_lr, clip_norm=5.0):
        self.optimizer = optimizer(initial_lr)
        grads, vars = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
        self.train_op = self.optimizer.apply_gradients(zip(gradients, vars))





