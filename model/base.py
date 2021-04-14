# coding:utf-8
import numpy as np
np.random.seed(12345)
import tensorflow as tf
tf.set_random_seed(12345)
import logging
import numpy as np
import os
from collections import OrderedDict, defaultdict
from train.du_trainer import Trainer

class BaseModel(object):
    def __init__(self, vocab=None,use_xla=False,label_map =None):
        self.vocab = vocab
        self.label_map =label_map 

        sess_conf = tf.ConfigProto()
        if use_xla:
            sess_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess_conf.gpu_options.allow_growth = True
        self.session = tf.Session(config=sess_conf)
        self.initialized = False

    def __del__(self):
        self.session.close()

    def load(self, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        logging.info('Loading model from %s' % path)
        saver = tf.train.Saver(var_list)
        checkpoint_path = tf.train.latest_checkpoint(path)
        saver.restore(self.session, save_path=checkpoint_path)
        self.initialized = True

    def save(self, path, global_step=None, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.save(self.session, path, global_step=global_step)

    def _build_graph(self):
        raise NotImplementedError

    def compile(self, *input):
        raise NotImplementedError

    def train_and_evaluate(self, *input):
        raise NotImplementedError

    def evaluate(self, *input):
        raise NotImplementedError

    # def inference(self, *input):
    #     raise NotImplementedError

    def train_and_evaluate(self, data_reader, evaluator, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10,batch_size=32):
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())

        Trainer._train_and_evaluate(self, data_reader, evaluator, epochs=epochs,
                                    eposides=eposides,
                                    save_dir=save_dir, summary_dir=summary_dir, save_summary_steps=save_summary_steps,batch_size=batch_size)

    def inference(self,data_reader,inference_batch_size =64,output_path=None,category_name=None):
        pad_id = 0
        eval_batches = data_reader.gen_mini_batches('test', inference_batch_size, pad_id, shuffle=False)

        Trainer._test_sess(self,eval_batches,label_map =self.label_map,output_path=output_path,category_name='category_name' if category_name is None else category_name)

    def evaluate(self,data_reader,eval_batch_size=40):
        pad_id = 0
