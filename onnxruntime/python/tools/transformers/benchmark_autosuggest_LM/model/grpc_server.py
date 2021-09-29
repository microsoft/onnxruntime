#####################
#### DO NOT EDIT ####
#####################

"""
owner: isst.

This file will start a grpc server 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import generic_serving_inference_pb2
import grpc
import utils
from concurrent import futures
from model import ModelImp

work_num = 2

model_imp = None
class GenericServer(generic_serving_inference_pb2.BetaGenericServiceServicer):
    def __init__(self):
        self.model = ModelImp()

    def Eval(self, request, context):
        qargs = request.data
        if (not qargs):
            raise Exception("empty request")

        response = generic_serving_inference_pb2.GenericResponse()
        response.data = model_imp.Eval(qargs)
        return response

    # TODO: implement EvalBinary once Grpc and Protobuf has been upgraded in base Docker images

def start(model):
    global model_imp
    model_imp = model
    listeningPort = utils.get_listening_port(9000)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=work_num))
    generic_serving_inference_pb2.add_GenericServiceServicer_to_server(GenericServer(), server)
    server.add_insecure_port("[::]:{port}".format(port = listeningPort))
    server.start()
    print("running")
    try:
        while True:
            time.sleep(999999)
    except KeyboardInterrupt:
        server.stop(0)
 
if __name__ == "__main__":
    model = ModelImp()
    start(model)
