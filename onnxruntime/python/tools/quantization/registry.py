from .quant_utils import QuantizationMode
from .operators.base_operator import QuantOperatorBase
from .operators.qdq_base_operator import QDQOperatorBase
from .operators.matmul import MatMulInteger, QLinearMatMul
from .operators.attention import AttentionQuant
from .operators.embed_layernorm import EmbedLayerNormalizationQuant
from .operators.gather import GatherQuant
from .operators.conv import QLinearConv, ConvInteger, QDQConv
from .operators.activation import QLinearActivation, QDQRemovableActivation
from .operators.binary_op import QLinearBinaryOp
from .operators.maxpool import QMaxPool
from .operators.gavgpool import QGlobalAveragePool
from .operators.lstm import LSTMQuant
from .operators.split import QSplit
from .operators.pad import QPad
from .operators.direct_q8 import Direct8BitOp
from .operators.resize import QResize

CommonOpsRegistry = {"Gather": GatherQuant,
                     "EmbedLayerNormalization": EmbedLayerNormalizationQuant,
                     "Reshape": Direct8BitOp,
                     "Transpose" : Direct8BitOp}

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
    "LeakyRelu": QLinearActivation,
    "Sigmoid": QLinearActivation,
    "MaxPool": QMaxPool,
    "GlobalAveragePool": QGlobalAveragePool,
    "Split": QSplit,
    "Pad": QPad,
    "Resize" : QResize,
}
QLinearOpsRegistry.update(CommonOpsRegistry)

QDQRegistry = {
    "Conv": QDQConv,
    "Clip": QDQRemovableActivation,
    "Relu": QDQRemovableActivation,
}


def CreateDefaultOpQuantizer(onnx_quantizer, node):
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer, node):
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps else QLinearOpsRegistry
    if node.op_type in registry.keys():
        return registry[node.op_type](onnx_quantizer, node)
    return QuantOperatorBase(onnx_quantizer, node)


def CreateQDQQuantizer(onnx_quantizer, node):
    if node.op_type in QDQRegistry.keys():
        return QDQRegistry[node.op_type](onnx_quantizer, node)
    return QDQOperatorBase(onnx_quantizer, node)
