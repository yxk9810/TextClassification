#! /usr/bin/env python
#-*- coding: utf-8 -*-
 
#####################################################
# Copyright (c) 2021 Sogou, Inc. All Rights Reserved
#####################################################
# File:    server.py
# Author:  root
# Date:    2021/05/14 11:35:16
# Brief:
#####################################################


from flask import Flask, request, jsonify
from flask import render_template

from tensorflow.python.keras.backend import set_session
import requests
import sys
sys.path.append('../')
import os 
app = Flask(__name__)
from data.vocab import Vocab
os.environ["CUDA_VISIBLE_DEVICES"] = " "
vocab_file = '../examples/politic_vocab5.txt'# vocab.load_from_file('vocab_bool.txt')
vocab = Vocab(lower=True)
from data.data_reader_new import DatasetReader
from model.text_cnn import TextCNN
if os.path.exists(vocab_file): vocab.load_from_file(vocab_file)
print(vocab.get_word_vocab())
@app.route('/')
def search_index():
    return render_template('index.html')

model = TextCNN (vocab,num_class=2,pretrained_word_embedding=vocab.embeddings,word_embedding_size=300)
model.load("/search/odin/jdwu/classification/cls_checkpoints/politic/best_weights")

@app.route('/get_politic_intent',methods=['POST','GET'])
def check_intent():
    if request.method == "POST":
        global model 
        data_reader = DatasetReader( use_pos_feature=False,
                             use_bert=False,
                             use_name_feature=False)
        query = request.form.get('input')
        if query.strip()=='':
            return jsonify({'message':'无效查询','status':0})

        data_reader.load_single_input(query)
        # print(vocab.get_char_vocab_size())
        data_reader.convert_to_ids(vocab)
        
     
        predict_label,sample = model.inference(data_reader,1)
        predict_label = predict_label[0]
        print("predict label "+str(predict_label)+'hhhh')
        print({'query':query,'status':str(1),'message':'涉政' if predict_label==1 else '非涉政查询'})
        return jsonify({'query':query,'is_politic':str(predict_label),'status':1,'message':'涉政' if predict_label==1 else '非涉政查询'})

    if request.method=='GET':
        data_reader = DatasetReader( use_pos_feature=False,
                             use_bert=False,
                             use_name_feature=False)
        query = request.args.get('query')
        data_reader.load_single_input(query)
        print(vocab.get_char_vocab_size())
        data_reader.convert_to_ids(vocab)
        
        model = TextCNN (vocab,num_class=2,pretrained_word_embedding=vocab.embeddings,word_embedding_size=300)
        model.load("/search/odin/jdwu/classification/cls_checkpoints/politic/best_weights")
        predict_label,sample = model.inference(data_reader,1)
        predict_label = predict_label[0]
        return jsonify({'query':request.args.get('query'),'is_politic':predict_label,'status':1,'message':'涉政' if predict_label==1 else '非涉政查询'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6001', debug=True, threaded = False)
 



















# vim: set expandtab ts=4 sw=4 sts=4 tw=100
