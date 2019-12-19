# encoding: utf-8
import os
import tensorflow as tf
import numpy as np
import re
import sys

from tensorflow.python.framework import ops

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8

check_point_regex = re.compile(r'^model_checkpoint_path:\s"(.*?)"$')

with tf.Session(config=config) as sess:
    model_dir_path = "./bert_news_checkpoint_new/best_weights"
    #with open(checkpoint_path, "r") as file:
    #    line = file.readline().strip()
    #    model_name = check_point_regex.match(line).group(1)
    meta_graph_name = "after-eposide-0.meta"
    meta_graph_path = os.path.join(model_dir_path, meta_graph_name)
    # 加载结构，模型参数和变量
    print ("importing meta graph:%s" % meta_graph_path)
    saver = tf.train.import_meta_graph(meta_graph_path)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir_path))
    graph = tf.get_default_graph()

    '''
    # 根据次数输出的变量名和操作名确定下边取值的名字
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print(v.name)

    for op in sess.graph.get_operations():
        print(op.name)
    '''
    token_ids = graph.get_tensor_by_name("token_ids:0")
    char_ids = graph.get_tensor_by_name("char_ids:0")
    intent_prob  = graph.get_tensor_by_name('probs:0')
    #output_na_prob2 = graph.get_tensor_by_name('Squeeze_8:0')

    print("begin saved_model...")
    save_dir_path = "distilled_intent_model_07232"
    export_model_version = 1
    builder = tf.saved_model.builder.SavedModelBuilder("./intent_model/%s" % export_model_version)
    inputs = {
		"token_ids": tf.saved_model.utils.build_tensor_info(token_ids),
		"char_ids": tf.saved_model.utils.build_tensor_info(char_ids),
    }

    # y 为最终需要的输出结果tensor
    outputs = {
               "intent_prob": tf.saved_model.utils.build_tensor_info(intent_prob),
               }


    #signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')
    #builder.add_meta_graph_and_variables(sess, ['test_saved_model'], {'test_signature': signature})

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs,
            outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={'serving_default': prediction_signature},
        legacy_init_op=legacy_init_op
    )

    builder.save()
    print("saved_model done..")
