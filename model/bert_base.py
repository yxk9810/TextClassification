# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import logging
from collections import OrderedDict, defaultdict
from model.base import BaseModel
from nn.layer import  BertEmbedding,Dropout
from libraries_albert import modeling_bak
from libraries_albert import optimization_finetuning as optimization
from train.du_trainer import Trainer



class BertBaseline(BaseModel):
    def __init__(self, vocab=None, bert_dir='', num_class=2,use_fp16=False,use_xla=False):
        super(BertBaseline, self).__init__(vocab,use_xla=use_xla)
        self.bert_dir = bert_dir
        self.num_class = num_class
        self.use_fp16 = use_fp16
        self._build_graph()

    def _build_graph(self):
        self.training = tf.placeholder_with_default(False, shape=(), name='is_training')
        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32,name='input_ids')
        self.input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32,name="input_mask")
        self.segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32,name="segment_ids")
        self.y = tf.placeholder(tf.int32, [None])
        self.bert_embedding = BertEmbedding(self.bert_dir)
        _,output_layer = self.bert_embedding(input_ids=self.input_ids, input_mask=self.input_mask,
                                           segment_ids=self.segment_ids, is_training=self.training,
                                           return_pool_output=True,use_fp16=self.use_fp16)


        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [self.num_class, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [self.num_class], initializer=tf.zeros_initializer())

        dropout = Dropout(0.9)
        output_layer = dropout(output_layer,self.training)
        # if is_training:
        #     # I.e., 0.1 dropout
        #     output_layer = tf.nn.dropout(output_layer, keep_prob=0.9,)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1,name="probs")
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(self.y, depth=self.num_class, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        self.loss = tf.reduce_mean(per_example_loss)
        self.probs = probabilities

        self.input_placeholder_dict = OrderedDict({
            "input_ids": self.input_ids,
            "segment_ids":self.segment_ids,
            "labels": self.y,
            "input_mask": self.input_mask,
            "training": self.training
        })

        self.output_variable_dict = OrderedDict({
            "predict": tf.argmax(self.probs, axis=1),
            "probabilities":probabilities
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

    def compile(self, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False):
        self.train_op = optimization.create_optimizer(self.loss, learning_rate, num_train_steps, num_warmup_steps,
                                                      use_tpu)

    def train_and_evaluate(self, data_reader, evaluator, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        if not self.initialized:
            self.bert_embedding.init_bert()
            self.session.run(tf.global_variables_initializer())

        Trainer._train_and_evaluate(self, data_reader, evaluator, epochs=epochs,
                                    eposides=eposides,
                                    save_dir=save_dir, summary_dir=summary_dir, save_summary_steps=save_summary_steps)


