import tensorflow as tf
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


#pip install pip install  tensorflow-serving-api==1.13.0
def main():
    max_seq_len = 40
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
    with tf.python_io.TFRecordWriter("tf_serving_warmup_requests") as writer:
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(name="intent_model", signature_name='serving_default'),
            inputs={
                "input_ids": tf.make_tensor_proto(input_ids, dtype=tf.int32,shape=[input_ids.shape[0], input_ids.shape[1]]),
                "input_mask": tf.make_tensor_proto(input_mask,dtype=tf.int32,shape=[input_mask.shape[0], input_mask.shape[1]]),
                "segment_ids": tf.make_tensor_proto(segment_ids,dtype=tf.int32, shape=[segment_ids.shape[0], segment_ids.shape[1]]),
                "training": tf.make_tensor_proto(False,dtype=tf.bool, shape=[])
            }
        )
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


if __name__ == "__main__":

    main()
