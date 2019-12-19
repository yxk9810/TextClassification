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
        self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.05)

        self.filter_sizes = [5,5,3,3,3,3]
        self.num_filters = 256

        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(self.vocab.get_char_vocab_size() + 1, self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)

        input_x = word_embedding(self.x)

        # pos_embedding = Embedding(pretrained_embedding=None,
        #                           embedding_shape=(self.pos_vocab_size, self.pos_embedding_size))
        #
        # input_x_pos = pos_embedding(self.pos_feature)
        #
        # feature_x = tf.one_hot(self.in_name_feature, depth=2)
        # input_x = tf.concat([input_x, feature_x], axis=-1)
        # input_x = tf.concat([input_x, input_x_pos], axis=-1)
        dropout = Dropout(self.keep_prob)
        input_x = dropout(input_x, self.training)

        input_x = tf.expand_dims(input_x,axis=-1)
        print(input_x.shape)

        # ============= Convolutional Layers =============
        with tf.name_scope("conv-maxpool-1"):
            conv1 = tf.layers.conv2d(
                input_x,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[0], self.word_embedding_size],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size=(3, 1),
                strides=(3, 1))
            pool1 = tf.transpose(pool1, [0, 1, 3, 2])

        with tf.name_scope("conv-maxpool-2"):
            conv2 = tf.layers.conv2d(
                pool1,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[1], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(
                conv2,
                pool_size=(3, 1),
                strides=(3, 1))
            pool2 = tf.transpose(pool2, [0, 1, 3, 2])

        with tf.name_scope("conv-3"):
            conv3 = tf.layers.conv2d(
                pool2,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[2], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv3 = tf.transpose(conv3, [0, 1, 3, 2])

        with tf.name_scope("conv-4"):
            conv4 = tf.layers.conv2d(
                conv3,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[3], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv4 = tf.transpose(conv4, [0, 1, 3, 2])

        with tf.name_scope("conv-5"):
            conv5 = tf.layers.conv2d(
                conv4,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[4], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv5 = tf.transpose(conv5, [0, 1, 3, 2])

        with tf.name_scope("conv-maxpool-6"):
            conv6 = tf.layers.conv2d(
                conv5,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[5], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool6 = tf.layers.max_pooling2d(
                conv6,
                pool_size=(3, 1),
                strides=(3, 1))
            pool6 = tf.transpose(pool6, [0, 2, 1, 3])
            print(pool6.get_shape().as_list())
            h_pool = tf.reshape(pool6, [-1, self.num_filters])
        print(h_pool.shape)



        print(input_x)
        fc1_layer = tf.keras.layers.Dense(128,activation=tf.nn.relu)
        fc1_out = fc1_layer(h_pool)
        # fc1_out = dropout(fc1_out,self.training)
        # fc2_layer = tf.keras.layers.Dense(1024,activation=tf.nn.relu)
        # fc2_out = fc2_layer(fc1_out)


        # encoder1 = BiLSTM(self.rnn_hidden_size,name='layer_1')
        # input_x,_ = encoder1(input_x,self.x_len)
        #
        # encoder2 = BiLSTM(self.rnn_hidden_size,name='layer_2')
        # input_x,_ = encoder2(input_x,self.x_len)
        # print(input_x.shape)
        # merge = tf.reshape(input_x,[tf.shape(input_x)[0],-1])

        # avg_pool = tf.reduce_mean(input_x,axis=1)
        # avg_max =  tf.reduce_max(input_x,axis=1)
        #
        # merge = tf.concat([avg_pool,avg_max],axis=1)
        # print(merge.shape)
        # h_conc_linear1 = tf.keras.layers.Dense(200,use_bias=False,activation=tf.nn.relu)(merge)
        # h_conc_linear2 = tf.keras.layers.Dense(200,use_bias=False,activation=tf.nn.relu)(merge)
        # merge = merge+h_conc_linear1+h_conc_linear2

        #
        # dense = tf.keras.layers.Dense(16,activation=tf.nn.relu)
        # merge = dense(merge)
        # merge = dropout(merge, self.training)
        self.logits = tf.keras.layers.Dense(self.num_class,activation=None)(fc1_out)
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





