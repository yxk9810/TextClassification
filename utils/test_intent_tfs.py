#coding:utf-8

import sys
import requests
import json
import time


def get_model_outputs(url='models/intent_model:predict',max_seq_len=40):
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

    batch_size = 1
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
            "token_ids": input_ids,
            "char_ids": input_masks,
        }
    }

    response = requests.post(url=url, data=json.dumps(data))
    outputs = json.loads(response.text)['outputs']
    return outputs


import argparse
import time


if __name__=='__main__':
    start = time.time()#print(get_model_outputs())
    start = time.time()#print(get_model_outputs())
    get_model_outputs()
    print("grpc cost :"+str((time.time()-start)/float(100)))#print(get_model_outputs())
    # for i in range(100):get_model_outputs(max_seq_len=300)
    # print("rest cost :"+str((time.time()-start)/float(100)))#print(get_model_outputs())
