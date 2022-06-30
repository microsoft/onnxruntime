from .operators.activation import QDQRemovableActivation, QLinearActivation
from .operators.argmax import QArgMax
from .operators.attention import AttentionQuant
from .operators.base_operator import QuantOperatorBase
from .operators.binary_op import QLinearBinaryOp
from .operators.concat import QDQConcat, QLinearConcat
from .operators.conv import ConvInteger, QDQConv, QLinearConv
from .operators.direct_q8 import Direct8BitOp, QDQDirect8BitOp
from .operators.embed_layernorm import EmbedLayerNormalizationQuant
from .operators.gather import GatherQuant
from .operators.gavgpool import QGlobalAveragePool
from .operators.gemm import QDQGemm, QLinearGemm
from .operators.lstm import LSTMQuant
from .operators.matmul import MatMulInteger, QDQMatMul, QLinearMatMul
from .operators.maxpool import QDQMaxPool, QMaxPool
from .operators.pad import QPad
from .operators.pooling import QLinearPool
from .operators.qdq_base_operator import QDQOperatorBase
from .operators.resize import QDQResize, QResize
from .operators.split import QSplit
from .quant_utils import QuantizationMode

CommonOpsRegistry = {
    "Gather": GatherQuant,
    "Transpose": Direct8BitOp,
    "EmbedLayerNormalization": EmbedLayerNormalizationQuant,
}

IntegerOpsRegistry = {
    "Conv": ConvInteger,
    "MatMul": MatMulInteger,
    "Attention": AttentionQuant,
    "LSTM": LSTMQuant,
}
IntegerOpsRegistry.update(CommonOpsRegistry)

QLinearOpsRegistry = {
    "ArgMax": QArgMax,
    "Conv": QLinearConv,
    "Gemm": QLinearGemm,
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
    "Reshape": Direct8BitOp,
    "Squeeze": Direct8BitOp,
    "Unsqueeze": Direct8BitOp,
    "Resize": QResize,
    "AveragePool": QLinearPool,
    "Concat": QLinearConcat,
}
QLinearOpsRegistry.update(CommonOpsRegistry)

QDQRegistry = {
    "Conv": QDQConv,
    "Gemm": QDQGemm,
    "Clip": QDQRemovableActivation,
    "Relu": QDQRemovableActivation,
    "Reshape": QDQDirect8BitOp,
    "Transpose": QDQDirect8BitOp,
    "Squeeze": QDQDirect8BitOp,
    "Unsqueeze": QDQDirect8BitOp,
    "Resize": QDQResize,
    "MaxPool": QDQMaxPool,
    "AveragePool": QDQDirect8BitOp,
    "Concat": QDQConcat,
    "MatMul": QDQMatMul,
}


def CreateDefaultOpQuantizer(onnx_quantizer, node):
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer, node):
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps else QLinearOpsRegistry
    if node.op_type in registry.keys():
        op_quantizer = registry[node.op_type](onnx_quantizer, node)
        if op_quantizer.should_quantize():
            return op_quantizer
    return QuantOperatorBase(onnx_quantizer, node)


def CreateQDQQuantizer(onnx_quantizer, node):
    if node.op_type in QDQRegistry.keys():
        return QDQRegistry[node.op_type](onnx_quantizer, node)
    return QDQOperatorBase(onnx_quantizer, node)
