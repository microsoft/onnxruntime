// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include "core/providers/common.h"

/**
 * This file defines mappings and structures to facilitate the translation of ONNX operations
 * and data types to their corresponding WebNN representations.
 *
 * It includes:
 * - Data type mappings between ONNX and WebNN.
 * - Lists of supported fallback integer types for WebNN.
 * - Decomposition of certain ONNX operations into sequences of WebNN operations.
 * - Structures and maps for input index-to-name translation for ONNX to WebNN ops.
 */
namespace onnxruntime {
namespace webnn {
const std::map<ONNX_NAMESPACE::TensorProto_DataType, std::string_view> onnx_to_webnn_data_type_map = {
    {ONNX_NAMESPACE::TensorProto_DataType_INT4, "int4"},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT4, "uint4"},
    {ONNX_NAMESPACE::TensorProto_DataType_BOOL, "uint8"},
    {ONNX_NAMESPACE::TensorProto_DataType_INT8, "int8"},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT8, "uint8"},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, "float16"},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, "float32"},
    {ONNX_NAMESPACE::TensorProto_DataType_INT32, "int32"},
    {ONNX_NAMESPACE::TensorProto_DataType_INT64, "int64"},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT32, "uint32"},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT64, "uint64"},
};

// This array contains the input/output data types of a WebNN graph that are allowed to be fallback to int32.
constexpr std::array<ONNX_NAMESPACE::TensorProto_DataType, 5> supported_fallback_integer_data_types = {
    ONNX_NAMESPACE::TensorProto_DataType_BOOL,
    ONNX_NAMESPACE::TensorProto_DataType_INT8,
    ONNX_NAMESPACE::TensorProto_DataType_UINT8,
    ONNX_NAMESPACE::TensorProto_DataType_UINT32,
    ONNX_NAMESPACE::TensorProto_DataType_INT64,
};

// Some ONNX ops are supported by decomposed WebNN ops.
// This map defines the relationship between ONNX ops and their corresponding decomposed ONNX ops.
// Use ONNX-to-ONNX op mapping to improve the search complexity for WebNN ops in the op_inputs_map.
const std::map<std::string_view, std::vector<std::string_view>> decomposed_op_map = {
    {"ConvInteger", {"Cast", "Conv", "DequantizeLinear"}},
    {"Einsum", {"MatMul", "Mul", "ReduceSum", "Reshape", "Transpose", "Trilu"}},
    {"GroupQueryAttention",
     {"Add", "Cast", "Concat", "CumSum", "Div", "Expand", "Less", "MatMul", "Reshape", "ScatterND",
      "Softmax", "Transpose", "Where"}},
    {"LRN", {"Add", "AveragePool", "Div", "Mul", "Pad", "Pow", "Transpose"}},
    {"MatMulInteger", {"Cast", "DequantizeLinear", "MatMul"}},
    {"MatMulNBits", {"Add", "DequantizeLinear", "MatMul", "Reshape", "Transpose"}},
    {"MultiHeadAttention", {"Add", "Cast", "Concat", "Div", "MatMul", "Reshape", "Softmax", "Transpose"}},
    {"RotaryEmbedding", {"Add", "Concat", "Gather", "Mul", "Reshape", "Slice", "Split"}},
    {"SimplifiedLayerNormalization", {"Add", "Div", "Mul", "Pow", "ReduceMean", "Sqrt"}},
    {"SkipSimplifiedLayerNormalization", {"Add", "Div", "Mul", "Pow", "ReduceMean", "Sqrt"}},
};

/**
 * Represents information about an input to a WebNN operation.
 *
 * This structure is used to map ONNX operation inputs to their corresponding
 * WebNN operation inputs. It contains the index of the input as specified
 * in the ONNX operation and the name of the input in the WebNN operation.
 *
 * InputInfo::index
 * The index of this input as defined in the ONNX operation specification.
 *
 * InputInfo::name
 * The name of this input in the WebNN operation.
 */
struct InputInfo {
  int index;
  std::string_view name;
};

struct WebnnOpInfo {
  std::string_view opType;
  std::vector<InputInfo> inputs;
  WebnnOpInfo(std::string_view op, std::initializer_list<InputInfo> in)
      : opType(op), inputs(in) {}
};

/**
 * Maps ONNX operation type to their corresponding WebNN operation type and input mappings.
 *
 * This unordered map provides a mapping between ONNX operation names (keys) and their corresponding
 * WebNN operation information (values). Each value is a `WebnnOpInfo` structure that contains:
 * - The WebNN operation name (`opType`).
 * - A vector of `InputInfo` structures, where each `InputInfo` specifies:
 *   - The index of the input in the ONNX operation (`index`).
 *   - The corresponding input name in the WebNN operation (`name`).
 *
 * For the ONNX operation "Abs", it has only one "input", which is at index 0 in the "Node.InputDefs" array.
 * The corresponding WebNN operation is "abs", and the input name is "input".
 *
 * This mapping is used to translate ONNX operations and their inputs into WebNN operations
 * and their respective input names.
 *
 * Order:
 *   The sorting rule is based on character length in ascending order (for better formatting),
 *   and for items with the same length, they are sorted alphabetically.
 */
const std::unordered_map<std::string_view, WebnnOpInfo> op_inputs_map = {
    {"Cos", {"cos", {{0, "input"}}}},
    {"Abs", {"abs", {{0, "input"}}}},
    {"Elu", {"elu", {{0, "input"}}}},
    {"Erf", {"erf", {{0, "input"}}}},
    {"Exp", {"exp", {{0, "input"}}}},
    {"Log", {"log", {{0, "input"}}}},
    {"Neg", {"neg", {{0, "input"}}}},
    {"Pad", {"pad", {{0, "input"}}}},
    {"Sin", {"sin", {{0, "input"}}}},
    {"Tan", {"tan", {{0, "input"}}}},
    {"Cast", {"cast", {{0, "input"}}}},
    {"Ceil", {"ceil", {{0, "input"}}}},
    {"Gelu", {"gelu", {{0, "input"}}}},
    {"Relu", {"relu", {{0, "input"}}}},
    {"Sign", {"sign", {{0, "input"}}}},
    {"Sqrt", {"sqrt", {{0, "input"}}}},
    {"Tanh", {"tanh", {{0, "input"}}}},
    {"Tile", {"tile", {{0, "input"}}}},
    {"Clip", {"clamp", {{0, "input"}}}},
    {"Floor", {"floor", {{0, "input"}}}},
    {"Shape", {"slice", {{0, "input"}}}},
    {"Slice", {"slice", {{0, "input"}}}},
    {"Split", {"split", {{0, "input"}}}},
    {"Sub", {"sub", {{0, "a"}, {1, "b"}}}},
    {"Add", {"add", {{0, "a"}, {1, "b"}}}},
    {"ArgMax", {"argMax", {{0, "input"}}}},
    {"ArgMin", {"argMin", {{0, "input"}}}},
    {"Div", {"div", {{0, "a"}, {1, "b"}}}},
    {"Expand", {"expand", {{0, "input"}}}},
    {"Max", {"max", {{0, "a"}, {1, "b"}}}},
    {"Min", {"min", {{0, "a"}, {1, "b"}}}},
    {"Mul", {"mul", {{0, "a"}, {1, "b"}}}},
    {"Pow", {"pow", {{0, "a"}, {1, "b"}}}},
    {"Concat", {"concat", {{0, "inputs"}}}},
    {"Not", {"logicalNot", {{0, "a"}}}},
    {"Flatten", {"reshape", {{0, "input"}}}},
    {"LpPool", {"l2Pool2d", {{0, "input"}}}},
    {"Reshape", {"reshape", {{0, "input"}}}},
    {"Sigmoid", {"sigmoid", {{0, "input"}}}},
    {"Softmax", {"softmax", {{0, "input"}}}},
    {"Squeeze", {"reshape", {{0, "input"}}}},
    {"Dropout", {"identity", {{0, "input"}}}},
    {"Trilu", {"triangular", {{0, "input"}}}},
    {"Equal", {"equal", {{0, "a"}, {1, "b"}}}},
    {"Identity", {"identity", {{0, "input"}}}},
    {"Less", {"lesser", {{0, "a"}, {1, "b"}}}},
    {"MaxPool", {"maxPool2d", {{0, "input"}}}},
    {"ReduceL1", {"reduceL1", {{0, "input"}}}},
    {"ReduceL2", {"reduceL2", {{0, "input"}}}},
    {"Resize", {"resample2d", {{0, "input"}}}},
    {"Softplus", {"softplus", {{0, "input"}}}},
    {"Softsign", {"softsign", {{0, "input"}}}},
    {"Unsqueeze", {"reshape", {{0, "input"}}}},
    {"Or", {"logicalOr", {{0, "a"}, {1, "b"}}}},
    {"HardSwish", {"hardSwish", {{0, "input"}}}},
    {"LeakyRelu", {"leakyRelu", {{0, "input"}}}},
    {"MatMul", {"matmul", {{0, "a"}, {1, "b"}}}},
    {"ReduceMax", {"reduceMax", {{0, "input"}}}},
    {"ReduceMin", {"reduceMin", {{0, "input"}}}},
    {"ReduceSum", {"reduceSum", {{0, "input"}}}},
    {"Transpose", {"transpose", {{0, "input"}}}},
    {"And", {"logicalAnd", {{0, "a"}, {1, "b"}}}},
    {"CumSum", {"cumulativeSum", {{0, "input"}}}},
    {"Xor", {"logicalXor", {{0, "a"}, {1, "b"}}}},
    {"GlobalLpPool", {"l2Pool2d", {{0, "input"}}}},
    {"Greater", {"greater", {{0, "a"}, {1, "b"}}}},
    {"Reciprocal", {"reciprocal", {{0, "input"}}}},
    {"ReduceMean", {"reduceMean", {{0, "input"}}}},
    {"Round", {"roundEven", {{0, "input"}}}},
    {"GlobalMaxPool", {"maxPool2d", {{0, "input"}}}},
    {"HardSigmoid", {"hardSigmoid", {{0, "input"}}}},
    {"ReduceProd", {"reduceProduct", {{0, "input"}}}},
    {"AveragePool", {"averagePool2d", {{0, "input"}}}},
    {"Gemm", {"gemm", {{0, "a"}, {1, "b"}, {2, "c"}}}},
    {"PRelu", {"prelu", {{0, "input"}, {1, "slope"}}}},
    {"ReduceLogSum", {"reduceLogSum", {{0, "input"}}}},
    {"Gather", {"gather", {{0, "input"}, {1, "indices"}}}},
    {"LessOrEqual", {"lesserOrEqual", {{0, "a"}, {1, "b"}}}},
    {"GlobalAveragePool", {"averagePool2d", {{0, "input"}}}},
    {"ReduceLogSumExp", {"reduceLogSumExp", {{0, "input"}}}},
    {"ReduceSumSquare", {"reduceSumSquare", {{0, "input"}}}},
    {"GatherND", {"gatherND", {{0, "input"}, {1, "indices"}}}},
    {"GreaterOrEqual", {"greaterOrEqual", {{0, "a"}, {1, "b"}}}},
    {"Conv", {"conv2d", {{0, "input"}, {1, "filter"}, {2, "bias"}}}},
    {"DynamicQuantizeLinear", {"dynamicQuantizeLinear", {{0, "input"}}}},
    {"GatherElements", {"gatherElements", {{0, "input"}, {1, "indices"}}}},
    {"ScatterND", {"scatterND", {{0, "input"}, {1, "indices"}, {2, "updates"}}}},
    {"Where", {"where", {{0, "condition"}, {1, "trueValue"}, {2, "falseValue"}}}},
    {"ConvTranspose", {"convTranspose2d", {{0, "input"}, {1, "filter"}, {2, "bias"}}}},
    {"QuantizeLinear", {"quantizeLinear", {{0, "input"}, {1, "scale"}, {2, "zeroPoint"}}}},
    {"ScatterElements", {"scatterElements", {{0, "input"}, {1, "indices"}, {2, "updates"}}}},
    {"LayerNormalization", {"layerNormalization", {{0, "input"}, {1, "scale"}, {2, "bias"}}}},
    {"DequantizeLinear", {"dequantizeLinear", {{0, "input"}, {1, "scale"}, {2, "zeroPoint"}}}},
    {"InstanceNormalization", {"instanceNormalization", {{0, "input"}, {1, "scale"}, {2, "bias"}}}},
    {"GRU", {"gru", {{0, "input"}, {1, "weight"}, {2, "recurrentWeight"}, {3, "bias"}, {5, "initialHiddenState"}}}},
    {"BatchNormalization", {"batchNormalization", {{0, "input"}, {1, "scale"}, {2, "bias"}, {3, "input_mean"}, {4, "input_var"}}}},
    {"LSTM", {"lstm", {{0, "input"}, {1, "weight"}, {2, "recurrentWeight"}, {3, "bias"}, {5, "initialHiddenState"}, {6, "initialCellState"}, {7, "peepholeWeight"}}}},
};
}  // namespace webnn
}  // namespace onnxruntime
