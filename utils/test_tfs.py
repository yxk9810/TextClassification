#coding:utf-8

import sys
import requests
import json
import time


def get_model_outputs(url='http://10.144.120.27:8531/v1/models/intent_model:predict',max_seq_len=40):
    input_id = [101, 6821, 3221, 671, 702, 3844, 6407, 4638, 1368, 2094, 102, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    segment_id = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if len(input_id) < max_seq_len:
        input_id.extend([0] * (max_seq_len - len(input_id)))
        input_mask.extend([0] * (max_seq_len - len(input_mask)))
        segment_id.extend([0] * (max_seq_len - len(segment_id)))

    batch_size = 10
    input_ids = []
    input_masks = []
    segment_ids = []
    for i in range(batch_size):
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)

    data = {
        "signature_name": "serving_default",
        "inputs": {
            "input_ids": input_ids,
            "input_mask": input_masks,
            "segment_ids": segment_ids,
        }
    }

    response = requests.post(url=url, data=json.dumps(data))
    print(response.text)
    outputs = json.loads(response.text)['outputs']
    print(outputs)
    return outputs


import argparse
import time

import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
max_grpc_time = 0
min_grpc_time = 0.050
def run(host, port, model, signature_name,max_seq_len=40):
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2.PredictionServiceStub(channel)

    start = time.time()

    # Call classification model to make prediction
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    input_id = [101, 6821, 3221, 671, 702, 3844, 6407, 4638, 1368, 2094, 102, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    segment_id = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if len(input_id) < max_seq_len:
        input_id.extend([0] * (max_seq_len - len(input_id)))
        input_mask.extend([0] * (max_seq_len - len(input_mask)))
        segment_id.extend([0] * (max_seq_len - len(segment_id)))

    batch_size = 10
    input_ids = []
    input_masks = []
    segment_ids = []
    for i in range(batch_size):
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    import numpy as np
    input_ids = np.array(input_ids)
    input_mask = np.array(input_masks)
    segment_ids = np.array(segment_ids)
    # print(input_ids.shape)
    # sys.exit(1)
    import tensorflow as tf

    request.inputs['input_ids'].CopyFrom(make_tensor_proto(input_ids, dtype=tf.int32,shape=[input_ids.shape[0], input_ids.shape[1]]))
    request.inputs['input_mask'].CopyFrom(make_tensor_proto(input_mask,dtype=tf.int32,shape=[input_mask.shape[0], input_mask.shape[1]]))
    request.inputs['segment_ids'].CopyFrom(make_tensor_proto(segment_ids,dtype=tf.int32, shape=[segment_ids.shape[0], segment_ids.shape[1]]))




    result = stub.Predict(request, 10.0)

    end = time.time()
    global max_grpc_time
    global min_grpc_time
    max_grpc_time = max(end - start,max_grpc_time)
    min_grpc_time = min(end-start,min_grpc_time)

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    #print(result)
    #print('time elapased: {}'.format(time_diff))


if __name__=='__main__':
    start = time.time()#print(get_model_outputs())
    # for i in range(100):run('10.144.120.27',port=8530,model='intent_model',signature_name='serving_default',max_seq_len=300)
    # print("grpc cost :"+str((time.time()-start)/float(100)))#print(get_model_outputs())
    # start = time.time()#print(get_model_outputs())
    for i in range(100):get_model_outputs(max_seq_len=40)
    # print("rest cost :"+str((time.time()-start)/float(100)))#print(get_model_outputs())
    # print(" max grpc time : "+str(max_grpc_time))
    # print("min grpc time "+str(min_grpc_time))
