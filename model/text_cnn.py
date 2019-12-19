#coding:utf-8
import numpy as np
np.random.seed(12345)
import tensorflow as tf
tf.set_random_seed(12345)
from model.base import BaseModel
from nn.layer import Embedding,Dropout

from collections import OrderedDict
from train.du_trainer import Trainer

class TextCNN(BaseModel):
    def __init__(self,vocab,pretrained_word_embedding=None,
             word_embedding_size=100,
             dropout_keep_prob=0.9,num_class=2,word_embedding_trainable=True,
             task_balance =1.0,
             soft_temperature=1
             ):
        super(TextCNN, self).__init__(vocab)
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
        self._build_graph()

    def _build_graph(self):
        self.x = tf.placeholder(tf.int32,[None,None])
        self.y = tf.placeholder(tf.int32,[None])
        self.domain = tf.placeholder(tf.int32,[None])
        print(self.x)
        # self.soft_target = tf.placeholder(tf.float32,[None,None])
        self.pos_feature = tf.placeholder(tf.int32,[None,None])
        # self.ask_word_feature = tf.placeholder(tf.int32,[None,None])
        # self.in_name_feature = tf.placeholder(tf.int32,[None,None])

        self.training = tf.placeholder_with_default(False,shape=(),name='is_training')

        word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                   embedding_shape=(self.vocab.get_word_vocab(), self.word_embedding_size),
                                   trainable=self.word_embedding_trainable)

        input_x = word_embedding(self.x)

        pos_embedding = Embedding(pretrained_embedding=None,
                                   embedding_shape=(self.pos_vocab_size, self.pos_embedding_size))

        input_x_pos = pos_embedding(self.pos_feature)
        #
        # feature_x = tf.one_hot(self.in_name_feature,depth=2)
        # ask_word_feature = tf.one_hot(self.ask_word_feature,depth=2)
        # input_x = tf.concat([input_x,feature_x],axis=-1)
        # # input_x = tf.concat([input_x,ask_word_feature],axis=-1)
        input_x = tf.concat([input_x,input_x_pos],axis=-1)
        # print(input_x.shape)
        dropout = Dropout(self.keep_prob)
        input_x = dropout(input_x,self.training)
        pooled =[]
        for idx,kernel_size in enumerate(self.filter_sizes1):
            con1d = tf.layers.conv1d(input_x,self.filter_nums1[idx],kernel_size,padding='same',activation=tf.nn.relu,
                                     name='conv1d-%d'%(idx))
            pooled_conv = tf.reduce_max(con1d,axis=1)
            pooled.append(pooled_conv)
        merge  = tf.concat(pooled,axis=1)
        merge = dropout(merge,self.training)
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

        self.loss = tf.reduce_mean(focal_loss_softmax(labels=self.y,logits=logits,alpha=0.5))
        #self.loss = tf.reduce_mean(focal_loss_softmax(self.y,logits))#tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.y))
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.y))

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
            "pos_feature":self.pos_feature,
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


