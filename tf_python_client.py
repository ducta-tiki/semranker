import numpy as np
import time
import tensorflow as tf

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


class SemRankerClient:
    def __init__(self):
        self.channel = implementations.insecure_channel('0.0.0.0', 9000)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

    def request(self, tensors):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'semranker'
        request.model_spec.signature_name = 'serving_default'

        for k, v in tensors.items():
            request.inputs[k].CopyFrom(tf.compat.v1.make_tensor_proto(v))
        tt = time.time()
        resp = self.stub.Predict(request, 3.0)
        v = list(resp.outputs['score'].float_val)
        # print(type(v))

        # print('total time: {}s'.format(time.time() - tt))
        return v
