import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum
from packaging import version

# TODO(kreeger): I don't know if I have to register an opset here.
#                See embed_layer_norm_gen.py for more details.

def GenerateInitializers():
    #
    # TODO - write me.
    #
    pass


def GenerateNodes(model_name, suffix=''):
    #
    # TODO - write me.
    #
    pass


def GenerateModel(model_name):
    #
    # TODO - write me.
    #
    pass


GenerateModel("test_model_1.onnx")
