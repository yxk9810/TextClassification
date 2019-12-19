# coding:utf-8

'''
code from https://www.kaggle.com/sergeykalutsky/introducing-bert-with-tensorflow

download bert-model
chinese :chinese_L-12_H-768_A-12
	bert_config.json
	bert_model.ckpt.data-00000-of-00001
	bert_model.ckpt.index
	bert_model.ckpt.meta
	vocab.txt
'''
import pandas as pd
import os
import numpy as np
import pandas as pd
import zipfile
import sys
import datetime

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

logging.getLogger("tensorflow").setLevel(logging.INFO)
import modeling
import optimization
import run_classifier
import tokenization
import tensorflow as tf

from sklearn.model_selection import train_test_split



def load_test_file(file=''):
    lines = [line.strip() for line in open(file,'r',encoding='utf-8').readlines()]
    return lines

def load_dataset(file='/search/odin/jdwu/intent_all_20w.csv'):
    texts = []
    labels = []
    with open(file, 'r', encoding='utf-8') as lines:
        for line in lines:
            data = line.strip().split('\t')
            label = 0 if len(data) == 1 else int(data[1])  # 1 #data = line.strip().split('\t')
            texts.append(data[0])
            labels.append(label)
    # texts = texts[:1000] #X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.1, random_state = 42)
    # labels = labels[:1000] #X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.1, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


trainX, testX, trainY, testY = load_dataset()




def create_examples(lines, set_type, labels=None):
    guid = set_type
    examples = []
    if set_type == 'train':
        for line, label in zip(lines, labels):
            text_a = line
            label = str(label)
            examples.append(run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    else:
        for line in lines:
            text_a = line
            label = '0'
            examples.append(run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    return examples


BERT_MODEL = '/search/odin/jdwu//chinese_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = '/search/odin/jdwu/chinese_L-12_H-768_A-12'
OUTPUT_DIR = './outputs20w'
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

# Model Hyper Parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 50
# Model configs
SAVE_CHECKPOINTS_STEPS = 500  # if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 100
NUM_TPU_CORES = 1
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')
print(DO_LOWER_CASE)

label_list = ['0', '1', '2']
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
train_examples = create_examples(trainX, 'train', labels=trainY)

# Train the model.
print('Please wait...')
print('covert example to features ')
train_features = run_classifier.convert_examples_to_features(
    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
print('***** Started training at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(len(train_examples)))
print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))

tpu_cluster_resolver = None  # Since training will happen on GPU, we won't need a cluster resolver
# TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
    use_one_hot_embeddings=True)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,  # If False training will fall on CPU or GPU, depending on what is available
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)

tf.logging.info("  Num steps = %d", num_train_steps)
train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('***** Finished training at {} *****'.format(datetime.datetime.now()))

eval_examples = create_examples(testX, 'train', labels=testY)

# print(train_examples)
# Train the model.
print('Please wait...')
print('covert example to features ')
eval_features = run_classifier.convert_examples_to_features(
    eval_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

print('***** Started evaluation at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(len(eval_examples)))
print('  Batch size = {}'.format(EVAL_BATCH_SIZE))

# Eval will be slightly WRONG on the TPU because it will truncate
# the last batch.
eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
eval_input_fn = run_classifier.input_fn_builder(
    features=eval_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=True)
result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
with tf.gfile.GFile(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
        print('  {} = {}'.format(key, str(result[key])))
        writer.write("%s = %s\n" % (key, str(result[key])))

result = estimator.predict(input_fn=eval_input_fn)
from tqdm import tqdm
import numpy as np

preds = []
labels = []
for prediction in tqdm(result):
    labels.append(prediction['labels'])  # for class_probability in prediction:
    preds.append(np.argmax(prediction['probabilities']))  # sys.exit(1)#for class_probability in prediction:
    # for class_probability in prediction:
    #  preds.append(float(class_probability))
# print(preds)
results = preds

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

print(accuracy_score(np.array(results), labels))
# print(f1_score(np.array(results), labels))
print(classification_report(labels, results, digits=4))

filename = '/search/odin/jdwu/tencent-1/sample_query_question_170w.csv'
test_lines = load_test_file(filename)
predict_examples = create_examples(test_lines,'test')
print('covert example to features ')
predict_features = run_classifier.convert_examples_to_features(
    predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

predict_input_fn =  run_classifier.input_fn_builder(
    features=predict_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=True)
result = estimator.predict(input_fn=eval_input_fn)
preds = []
for prediction in tqdm(result):
    preds.append(np.argmax(prediction['probabilities']))
print(len(predict_examples))
print(len(preds))
if len(predict_examples)==len(preds):
    writer = open('test_predict_result.txt','a+',encoding='utf-8')
    for idx,line in enumerate(predict_examples):
        writer.write(line.strip()+'\t'+preds[idx]+'\n')
    writer.close()
'''
writer = open('eval_predict.txt','a+')
testY= labels
for i in range(len(testY)): 
    writer.write(str(testX[i])+'\t'+results[i]+'\t'+testY[i])
writer.close()
'''
