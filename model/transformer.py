#coding:utf-8
import numpy as np
np.random.seed(12345)
import tensorflow as tf
tf.set_random_seed(12345)
from model.base import BaseModel
from nn.layer import Embedding,Dropout

from collections import OrderedDict
from train.du_trainer import Trainer

class Transformer(BaseModel):
    def __init__(self,vocab,pretrained_word_embedding=None,
             word_embedding_size=100,
             dropout_keep_prob=0.9,num_class=2,word_embedding_trainable=True,
             task_balance =1.0,
             soft_temperature=1,label_map=None
             ):
        super(TextCNN, self).__init__(vocab,label_map)
        self.filter_sizes1 = [2, 3, 4, 5, 6]
        self.filter_nums1 = [128, 128, 64, 64, 64]
        self.keep_prob = dropout_keep_prob
        self.num_class = num_class
        self.word_embedding_size =word_embedding_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.word_embedding_trainable = word_embedding_trainable
        self.pos_vocab_size = 56
        self.pos_embedding_size =12
        self.task_balance=task_balance
        self.softmax_temperature =soft_temperature
        self.num_blocks = 3 
        self._build_graph()

    def _build_graph(self):
        self.x = tf.placeholder(tf.int32,[None,None])
        self.y = tf.placeholder(tf.int32,[None])
        self.domain = tf.placeholder(tf.int32,[None])
        # self.soft_target = tf.placeholder(tf.float32,[None,None])
        self.pos_feature = tf.placeholder(tf.int32,[None,None])
        # self.ask_word_feature = tf.placeholder(tf.int32,[None,None])
        # self.in_name_feature = tf.placeholder(tf.int32,[None,None])

        self.training = tf.placeholder_with_default(False,shape=(),name='is_training')

        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(self.vocab.get_word_vocab(), self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)

        input_x = word_embedding(self.x)

        input_x +=self.positional_encoding(input_x,masking=False)
        self.enc = input_x 

        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                self.enc = self.multihead_attention(queries=self.enc,keys=self.enc,num_units=256,num_heads=8)
                self.enc = self.feedforward(self.enc,num_units=[4*256,256])
        input_x = self.enc 

        #
        # feature_x = tf.one_hot(self.in_name_feature,depth=2)
        # ask_word_feature = tf.one_hot(self.ask_word_feature,depth=2)
        # input_x = tf.concat([input_x,feature_x],axis=-1)
        # # input_x = tf.concat([input_x,ask_word_feature],axis=-1)
        # input_x = tf.concat([input_x,input_x_pos],axis=-1)
        # print(input_x.shape)
        dropout = Dropout(self.keep_prob)
        merge = self.input_x 
        # merge = tf.layers.batch_normalization(inputs=merge)
        # dense1 = tf.keras.layers.Dense(128,activation=tf.nn.tanh)
        merge = tf.layers.dense(merge,128,activation=tf.nn.tanh,name='dense1')
        # merge=tf.layers.batch_normalization(inputs=merge)
        merge = dropout(merge,self.training)
        logits = tf.layers.dense(merge,self.num_class,activation=None,use_bias=False)
        # logits = dense2(merge,name='dense2')
        self.prob = tf.nn.softmax(logits)

        domain_logits = tf.layers.dense(merge,2,activation=None,use_bias=False)
        self.domain_prob = tf.nn.softmax(domain_logits)
        # print(self.prob)

        from nn.loss import  softmax_with_logits_label_smooth

        from nn.loss import focal_loss_softmax

        #self.loss = tf.reduce_mean(focal_loss_softmax(labels=self.y,logits=logits,alpha=0.5))
        #self.loss = tf.reduce_mean(focal_loss_softmax(self.y,logits))#tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.y))
        #self.loss = tf.reduce_mean(softmax_with_logits_label_smooth(logits=logits,labels=self.y))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.y))

        # self.domain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=domain_logits,labels=self.domain))
        # self.loss+=self.loss + lossL2

        # self.soft_loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits/self.softmax_temperature,labels=self.soft_target)
        # )
        # self.task_balance=1.0
        # self.soft_loss =0.0
        # self.loss *=self.task_balance
        # self.loss += (1-self.task_balance)*self.soft_loss*(self.softmax_temperature**2)
        # self.loss +=self.domain_loss
        global_step = tf.train.get_or_create_global_step()

        self.input_placeholder_dict = OrderedDict({
            "token_ids": self.x,
            "labels":self.y,
            # "domain":self.domain,
            # 'soft_target':self.soft_target,
            # "features":self.in_name_feature,
            # "pos_feature":self.pos_feature,
            # 'ask_word_feature':self.ask_word_feature,
            "training": self.training,
        })

        self.output_variable_dict = OrderedDict({
            "predict": tf.argmax(logits,axis=1),
            "prob":self.prob

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

    def compile(self, optimizer, initial_lr,clip_norm=5.0):
        self.optimizer = optimizer(initial_lr)
        grads, vars = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(grads,clip_norm=clip_norm)
        self.train_op = self.optimizer.apply_gradients(zip(gradients, vars))


    def evaluate(self, batch_generator, evaluator):

        Trainer._evaluate(self, batch_generator, evaluator)

    def positional_encoding(inputs,maxlen,masking=True,scope="positional_encoding"):
        E = inputs.get_shape().as_list()[-1] # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

            return tf.to_float(outputs)


    def multihead_attention(self, queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None):
        '''Applies multihead attention.
        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate)

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs

    def feedforward(self, inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    reuse=None):
        '''Point-wise feed forward net.
        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.normalize(outputs)

        return outputs

    def normalize(self, inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs



