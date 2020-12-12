from .quant_utils import QuantizationMode
from .operators.base_operator import QuantOperatorBase
from .operators.matmul import MatMulInteger, QLinearMatMul
from .operators.attention import AttentionQuant
from .operators.embed_layernorm import EmbedLayerNormalizationQuant
from .operators.gather import GatherQuant
from .operators.conv import QLinearConv, ConvInteger
from .operators.activation import QLinearActivation
from .operators.binary_op import QLinearBinaryOp
from .operators.maxpool import QMaxPool
from .operators.gavgpool import QGlobalAveragePool
from. operators.lstm import LSTMQuant

CommonOpsRegistry = {"Gather": GatherQuant, "EmbedLayerNormalization": EmbedLayerNormalizationQuant}

IntegerOpsRegistry = {
    "Conv": ConvInteger,
    "MatMul": MatMulInteger,
    "Attention": AttentionQuant,
    "LSTM": LSTMQuant,
}
IntegerOpsRegistry.update(CommonOpsRegistry)

QLinearOpsRegistry = {
    "Conv": QLinearConv,
    "MatMul": QLinearMatMul,
    "Add": QLinearBinaryOp,
    "Mul": QLinearBinaryOp,
    "Relu": QLinearActivation,
    "Clip": QLinearActivation,
    "LeakyRelu" : QLinearActivation,
    "Sigmoid" : QLinearActivation,
    "MaxPool": QMaxPool,
    "GlobalAveragePool": QGlobalAveragePool,
}
QLinearOpsRegistry.update(CommonOpsRegistry)


def CreateDefaultOpQuantizer(onnx_quantizer, node):
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer, node):
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps else QLinearOpsRegistry
    if node.op_type in registry.keys():
        return registry[node.op_type](onnx_quantizer, node)
    return QuantOperatorBase(onnx_quantizer, node)
