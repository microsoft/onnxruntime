// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <unordered_set>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <set>

#include "../backend_utils.h"
#include "../backend_manager.h"
#include "../ov_interface.h"
#include "data_ops.h"
#include "capability.h"
#include "utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245 5208)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
// #include <ngraph/ngraph.hpp>
// #include <ngraph/frontend/onnx_import/onnx.hpp>
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace openvino_ep {

// Ops which are supported only in models(as intermediate nodes) and not in unit tests
std::set<std::string> ops_supported_only_in_model = {
    "Add",
    "Cast",
    "Celu",
    "Concat",
    "ConstantOfShape",
    "DequantizeLinear",
    "Dropout",
    "Einsum",
    "Exp",
    "Expand",
    "EyeLike",
    "GatherElements",
    "GatherND",
    "GridSample",
    "Identity",
    "LayerNormalization",
    "Loop",
    "LSTM",
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "OneHot",
    "Pad",
    "QuantizeLinear",
    "RandomNormalLike",
    "Range",
    "ReduceMin",
    "Resize",
    "Round",
    "Shape",
    "Slice",
    "Split",
    "Tile",
    "TopK",
    "Trilu"};

// Ops which are supported as functions (as composite ops)
std::set<std::string> ops_supported_as_function = {
    "LessOrEqual",
    "GreaterOrEqual",
    "LayerNormalization",
    "Celu"};

std::vector<SupportedOp> supported_op_mode = {
    {"Abs", V_2020_4, {"CPU", "GPU"}},
    {"Acos", V_2020_4, {"CPU"}},
    {"Acos", V_2022_1, {"GPU"}},
    {"Acosh", V_2020_4, {"CPU"}},
    {"Acosh", V_2022_1, {"GPU"}},
    {"Add", V_2020_4, {"CPU", "GPU"}},
    {"And", V_2020_4, {"CPU", "GPU"}},
    {"ArgMax", V_2020_4, {"CPU"}},
    {"ArgMax", V_2021_1, {"GPU"}},
    {"ArgMin", V_2020_4, {"CPU"}},
    {"ArgMin", V_2022_1, {"GPU"}},
    {"Asin", V_2020_4, {"CPU", "GPU"}},
    {"Asinh", V_2020_4, {"CPU", "GPU"}},
    {"Atan", V_2020_4, {"CPU", "GPU"}},
    {"Atanh", V_2020_4, {"CPU"}},
    {"Atanh", V_2022_1, {"GPU"}},
    {"AveragePool", V_2020_4, {"CPU", "GPU"}},
    {"BatchNormalization", V_2020_4, {"CPU", "GPU"}},
    {"BitShift", V_2022_1, {"CPU"}},
    {"Cast", V_2020_4, {"CPU", "GPU"}},
    {"CastLike", V_2023_1, {"CPU", "GPU"}},
    {"Ceil", V_2020_4, {"GPU"}},
    {"Ceil", V_2021_4, {"CPU"}},
    {"Celu", V_2022_1, {"CPU", "GPU"}},
    {"Clip", V_2020_4, {"CPU", "GPU"}},
    {"Compress", V_2023_1, {"CPU", "GPU"}},
    {"Concat", V_2020_4, {"CPU", "GPU"}},
    {"Constant", V_2020_4, {"CPU", "GPU"}},
    {"ConstantOfShape", V_2020_4, {"CPU", "GPU"}},
    {"Conv", V_2020_4, {"CPU", "GPU"}},
    {"ConvInteger", V_2022_1, {"CPU", "GPU"}},
    {"ConvTranspose", V_2020_4, {"CPU", "GPU"}},
    {"Cos", V_2020_4, {"CPU"}},
    {"Cos", V_2022_1, {"GPU"}},
    {"Cosh", V_2020_4, {"CPU"}},
    {"Cosh", V_2022_1, {"GPU"}},
    {"CumSum", V_2022_1, {"CPU", "GPU"}},
    {"DepthToSpace", V_2020_4, {"CPU", "GPU"}},
    {"DequantizeLinear", V_2021_4, {"CPU", "GPU"}},
    {"Div", V_2020_4, {"CPU", "GPU"}},
    {"Dropout", V_2020_4, {"CPU", "GPU"}},
    {"Elu", V_2020_4, {"CPU", "GPU"}},
    {"Einsum", V_2023_1, {"CPU", "GPU"}},
    {"EPContext", V_2024_0, {"CPU", "GPU", "NPU"}},
    {"Equal", V_2020_4, {"CPU", "GPU"}},
    {"Erf", V_2020_4, {"CPU", "GPU"}},
    {"Exp", V_2020_4, {"CPU", "GPU"}},
    {"Expand", V_2022_1, {"CPU", "GPU"}},
    {"EyeLike", V_2022_1, {"CPU"}},
    {"Flatten", V_2020_4, {"CPU", "GPU"}},
    {"Floor", V_2020_4, {"CPU", "GPU"}},
    {"Gather", V_2020_4, {"CPU", "GPU"}},
    {"GatherElements", V_2022_2, {"CPU", "GPU"}},
    {"GatherND", V_2021_4, {"CPU", "GPU"}},
    {"Gelu", V_2023_1, {"CPU", "GPU"}},
    {"Gemm", V_2020_4, {"CPU", "GPU"}},
    {"GlobalAveragePool", V_2020_4, {"CPU", "GPU"}},
    {"GlobalLpPool", V_2020_4, {"CPU", "GPU"}},
    {"GlobalMaxPool", V_2022_1, {"CPU", "GPU"}},
    {"Greater", V_2020_4, {"CPU", "GPU"}},
    {"GreaterOrEqual", V_2022_1, {"CPU", "GPU"}},
    {"GridSample", V_2022_3, {"CPU"}},
    {"GridSample", V_2023_0, {"GPU"}},
    {"HardMax", V_2023_1, {"CPU", "GPU"}},
    {"Identity", V_2020_4, {"CPU", "GPU"}},
    {"If", V_2022_3, {"CPU", "GPU"}},
    {"ImageScaler", V_2022_1, {"CPU", "GPU"}},
    {"InstanceNormalization", V_2020_4, {"CPU", "GPU"}},
    {"HardSigmoid", V_2020_4, {"CPU", "GPU"}},
    {"HardMax", V_2022_1, {"CPU", "GPU"}},
    {"LayerNormalization", V_2023_0, {"CPU", "GPU"}},
    {"LeakyRelu", V_2020_4, {"CPU", "GPU"}},
    {"Less", V_2020_4, {"CPU", "GPU"}},
    {"LessOrEqual", V_2022_1, {"CPU", "GPU"}},
    {"Log", V_2020_4, {"CPU", "GPU"}},
    {"LogSoftMax", V_2022_1, {"CPU", "GPU"}},
    {"Loop", V_2021_4, {"CPU", "GPU"}},
    {"LpNormalization", V_2023_1, {"CPU", "GPU"}},
    {"LRN", V_2020_4, {"CPU", "GPU"}},
    {"LSTM", V_2020_4, {"CPU", "GPU"}},
    {"MatMul", V_2020_4, {"CPU", "GPU"}},
    {"MatMulInteger", V_2022_1, {"CPU"}},
    {"Max", V_2020_4, {"CPU", "GPU"}},
    {"MaxPool", V_2020_4, {"CPU", "GPU"}},
    {"Mean", V_2020_4, {"CPU", "GPU"}},
    {"MeanVarianceNormalization", V_2022_1, {"CPU", "GPU"}},
    {"Min", V_2020_4, {"CPU", "GPU"}},
    {"Mod", V_2022_1, {"CPU", "GPU"}},
    {"Mul", V_2020_4, {"CPU", "GPU"}},
    {"Neg", V_2020_4, {"CPU", "GPU"}},
    {"NonMaxSuppression", V_2021_1, {"CPU", "GPU"}},
    {"NonZero", V_2021_1, {"CPU"}},
    {"NonZero", V_2023_0, {"GPU"}},
    {"Not", V_2021_1, {"CPU", "GPU"}},
    {"Not", V_2020_4, {"CPU", "GPU"}},
    {"OneHot", V_2020_4, {"CPU", "GPU"}},
    {"Or", V_2022_1, {"CPU", "GPU"}},
    {"Pad", V_2020_4, {"CPU", "GPU"}},
    {"Pow", V_2020_4, {"CPU", "GPU"}},
    {"PRelu", V_2020_4, {"CPU", "GPU"}},
    {"QLinearMatMul", V_2022_3, {"CPU"}},
    {"QuantizeLinear", V_2021_4, {"CPU", "GPU"}},
    {"RNN", V_2023_1, {"CPU", "GPU"}},
    {"RandomNormalLike", V_2023_0, {"CPU", "GPU"}},
    {"RandomNormalLike", V_2023_0, {"CPU", "GPU"}},
    {"RandomNormal", V_2023_0, {"CPU", "GPU"}},
    {"Range", V_2022_1, {"CPU", "GPU"}},
    {"Reciprocal", V_2020_4, {"CPU", "GPU"}},
    {"ReduceL1", V_2022_1, {"CPU", "GPU"}},
    {"ReduceL2", V_2022_1, {"CPU", "GPU"}},
    {"ReduceLogSum", V_2020_4, {"CPU"}},
    {"ReduceLogSum", V_2022_1, {"CPU", "GPU"}},
    {"ReduceLogSumExp", V_2022_1, {"CPU", "GPU"}},
    {"ReduceMax", V_2020_4, {"CPU", "GPU"}},
    {"ReduceMean", V_2020_4, {"CPU", "GPU"}},
    {"ReduceMin", V_2020_4, {"CPU", "GPU"}},
    {"ReduceProd", V_2020_4, {"CPU"}},
    {"ReduceProd", V_2022_1, {"GPU"}},
    {"ReduceSum", V_2020_4, {"CPU", "GPU"}},
    {"ReduceSumSquare", V_2020_4, {"CPU"}},
    {"ReduceSumSquare", V_2022_1, {"CPU", "GPU"}},
    {"Relu", V_2020_4, {"CPU", "GPU"}},
    {"Resize", V_2020_4, {"CPU"}},
    {"Resize", V_2022_1, {"GPU"}},
    {"Reshape", V_2020_4, {"CPU", "GPU"}},
    {"ReverseSequence", V_2022_1, {"CPU", "GPU"}},
    {"RoiAlign", V_2021_1, {"CPU", "GPU"}},
    {"Round", V_2021_4, {"CPU", "GPU"}},
    {"Scatter", V_2022_1, {"CPU", "GPU"}},
    {"ScatterElements", V_2022_1, {"CPU", "GPU"}},
    {"ScatterND", V_2022_1, {"CPU", "GPU"}},
    {"Selu", V_2020_4, {"CPU", "GPU"}},
    {"Shape", V_2020_4, {"CPU", "GPU"}},
    {"Shrink", V_2022_1, {"CPU", "GPU"}},
    {"Sigmoid", V_2020_4, {"CPU", "GPU"}},
    {"Sign", V_2020_4, {"CPU"}},
    {"Sign", V_2022_1, {"GPU"}},
    {"Sin", V_2022_1, {"CPU", "GPU"}},
    {"Sinh", V_2020_4, {"CPU"}},
    {"Size", V_2022_1, {"CPU", "GPU"}},
    {"Slice", V_2020_4, {"CPU", "GPU"}},
    {"Softmax", V_2020_4, {"CPU", "GPU"}},
    {"Softplus", V_2022_1, {"CPU", "GPU"}},
    {"Softsign", V_2022_1, {"CPU", "GPU"}},
    {"SpaceToDepth", V_2020_4, {"CPU", "GPU"}},
    {"Split", V_2020_4, {"CPU", "GPU"}},
    {"Sqrt", V_2020_4, {"CPU", "GPU"}},
    {"Squeeze", V_2020_4, {"CPU", "GPU"}},
    {"Softsign", V_2020_4, {"CPU"}},
    {"Sub", V_2020_4, {"CPU", "GPU"}},
    {"Sum", V_2020_4, {"CPU", "GPU"}},
    {"Tan", V_2020_4, {"CPU", "GPU"}},
    {"Tanh", V_2020_4, {"CPU", "GPU"}},
    {"ThresholdedRelu", V_2022_1, {"CPU", "GPU"}},
    {"Tile", V_2021_3, {"CPU", "GPU"}},
    {"Transpose", V_2020_4, {"CPU", "GPU"}},
    {"Trilu", V_2023_0, {"CPU", "GPU"}},
    {"TopK", V_2020_4, {"CPU", "GPU"}},
    {"Upsample", V_2020_4, {"CPU", "GPU"}},
    {"Unsqueeze", V_2020_4, {"CPU", "GPU"}},
    {"Where", V_2022_1, {"CPU", "GPU"}},
    {"Xor", V_2022_1, {"CPU", "GPU"}},
};

const OrtGraphApi* DataOps::graph_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetGraphApi(ORT_API_VERSION);

void DataOps::populate_types_supported() {
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
  supported_types_initializer_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16));
  supported_types_initializer_.insert(
      std::make_pair(V_2021_1, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  supported_types_initializer_.insert(
      std::make_pair(V_2021_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8));
  supported_types_initializer_.insert(
      std::make_pair(V_2021_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8));

  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));
  supported_types_npu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
  supported_types_npu_.insert(
      std::make_pair(V_2021_1, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));

  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8));
  supported_types_cpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
  supported_types_cpu_.insert(
      std::make_pair(V_2022_2, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));

  supported_types_gpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  supported_types_gpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));
  supported_types_gpu_.insert(
      std::make_pair(V_2020_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
  supported_types_gpu_.insert(
      std::make_pair(V_2021_1, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  supported_types_gpu_.insert(
      std::make_pair(V_2021_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8));
  supported_types_gpu_.insert(
      std::make_pair(V_2021_4, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8));
  supported_types_gpu_.insert(
      std::make_pair(V_2022_1, ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
}

void DataOps::populate_op_mode_supported() {
  no_dimension_supported_.push_back({"Add", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"And", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"Cast", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Ceil", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Clip", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"Div", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"DequantizeLinear", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Equal", V_2022_1, {"CPU"}});
  no_dimension_supported_.push_back({"Equal", V_2023_0, {"GPU"}});
  no_dimension_supported_.push_back({"Expand", V_2023_3, {"CPU"}});
  no_dimension_supported_.push_back({"Floor", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Gather", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Identity", V_2023_0, {"All"}});
  no_dimension_supported_.push_back({"Less", V_2022_1, {"CPU"}});
  no_dimension_supported_.push_back({"Loop", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Min", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Mul", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Neg", V_2023_0, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"Pow", V_2023_0, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"QuantizeLinear", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Range", V_2021_2, {"All"}});
  no_dimension_supported_.push_back({"ReduceMax", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"ReduceMin", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"ReduceProd", V_2022_1, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"Reshape", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"Shape", V_2022_1, {"GPU"}});
  no_dimension_supported_.push_back({"Shape", V_2023_0, {"CPU"}});
  no_dimension_supported_.push_back({"Sqrt", V_2023_0, {"All"}});
  no_dimension_supported_.push_back({"Squeeze", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Sub", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Unsqueeze", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Where", V_2021_2, {"All"}});

  subgraph_supported_.push_back({"Cast", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Concat", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Div", V_2021_1, {"CPU"}});
  subgraph_supported_.push_back({"Gather", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Identity", V_2021_1, {"CPU"}});
  subgraph_supported_.push_back({"Mul", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Sub", V_2021_1, {"CPU"}});
  subgraph_supported_.push_back({"Transpose", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Unsqueeze", V_2020_4, {"All"}});

  // populate unsupportedmode_t
  {
    UnsupportedOpMode obj = {{V_2024_1},
                             [this](const OrtNode* node) {
                               size_t num_input = 0;
                               graph_api_->OrtNode_GetNumInputs(node, &num_input);
                               // If the Input of ReduceMax op is UINT8, it is rejected (Due to output mismatch)
                               for (size_t i = 0; i < num_input; i++) {
                                 const char* input_name = nullptr;
                                 graph_api_->OrtNode_GetIthInputName(node, i, &input_name);
                                 OrtValueInfoRef* value_info = nullptr;
                                 graph_api_->OrtGraph_GetValueInfo(graph_viewer_, input_name, &value_info);
                                 ONNXTensorElementDataType dtype = value_info->data_type;
                                 graph_api_->OrtGraph_ReleaseValueInfo(value_info);
                                 if (dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                                     dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"ReduceMax", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2023_1, V_2023_2, V_2023_3, V_2024_0, V_2024_1},
                             [this](const OrtNode* node) {
                               const char* input1_name = nullptr;
                               graph_api_->OrtNode_GetIthInputName(node, 1, &input1_name);
                               OrtValueInfoRef* value_info = nullptr;
                               graph_api_->OrtGraph_GetValueInfo(graph_viewer_, input1_name, &value_info);
                               if (value_info->shape != nullptr) {
                                for (int i = 0; i < value_info->shape_len; i++) {
                                  if (value_info->shape[i] == 0) {
                                    graph_api_->OrtGraph_ReleaseValueInfo(value_info);
                                    return true;
                                  }
                                }
                               }

                               graph_api_->OrtGraph_ReleaseValueInfo(value_info);
                               return false;
                             }};
    op_list_.insert({"Reshape", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2023_1, V_2023_2, V_2023_3, V_2024_0, V_2024_1},
                             [this](const OrtNode* node) {
                               // If the operator is unsqueeze
                               // If axes is an input, then we cannot produce a static graph.
                               // Conversion fails in convert_function_to_cnn_network.
                               size_t num_input = 0;
                               graph_api_->OrtNode_GetNumInputs(node, &num_input);
                               for (size_t i = 0; i < num_input; i++) {
                                 const char* input_name = nullptr;
                                 graph_api_->OrtNode_GetIthInputName(node, i, &input_name);
                                 if (!strcmp(input_name, "axes")) {
                                   return true;
                                 }
                               }
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"Unsqueeze", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2023_1, V_2023_2, V_2023_3, V_2024_0, V_2024_1},
                             [this](const OrtNode* node) {
                               // check for attributes
                               size_t key_count = 0;
                               graph_api_->OrtNode_GetAttributeKeyCount(node, "scales", &key_count);
                               if (key_count > 0) {
                                 int float_size = 0;
                                 graph_api_->OrtNode_GetAttributeFloatSize(node, "scales", &float_size);
                                 if (float_size > 2) {
                                   float f0, f1;
                                   graph_api_->OrtNode_GetAttributeIthFloat(node, "scales", 0, &f0);
                                   graph_api_->OrtNode_GetAttributeIthFloat(node, "scales", 1, &f1);
                                   if (f0 != 1.f || f1 != 1.f) return true;
                                 }
                               }

                               // check for input dimensions
                               const char* input0_name = nullptr;
                               graph_api_->OrtNode_GetIthInputName(node, 0, &input0_name);
                               OrtValueInfoRef* value_info = nullptr;
                               graph_api_->OrtGraph_GetValueInfo(graph_viewer_, input0_name, &value_info);
                               if (value_info->shape != nullptr) {
                                 if (value_info->shape_len == 1 || value_info->shape_len == 4) {
                                   graph_api_->OrtGraph_ReleaseValueInfo(value_info);
                                   return true;
                                 }
                               }
                               // x_arg supports only float, int8 and float16 type
                               ONNXTensorElementDataType dtype = value_info->data_type;
                               graph_api_->OrtGraph_ReleaseValueInfo(value_info);
                               if (dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                                   dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                                   dtype == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                                 return false;
                               return true;
                             }};
    op_list_.insert({"Upsample", obj});
  }
}

bool DataOps::op_is_supported(std::string name, std::vector<SupportedOp>& op_list) {
  bool auto_support = false;
  bool multi_support = false;
  for (size_t i = 0; i < op_list.size(); i++) {
    if (op_list[i].optype == name) {
      if (op_list[i].version <= version_id_) {
        auto it = op_list[i].device_type.begin();
        while (it != op_list[i].device_type.end()) {
          // status variable is set to True if it's Hetero/Multi/Auto device type
          bool status = false;

          // The operator to be marked true, it should be supported by either of the devices specified with HETERO
          if (device_id_.find("HETERO") == 0) {
            status = true;
            if (device_id_.find(*it) != std::string::npos || (*it == "All")) {
              return true;
            }
          }

          // The operator to be marked true, it should be supported by all the devices specified with MULTI/AUTO
          if (device_id_.find("MULTI") == 0) {
            status = true;
            if ((*it == "All") || device_id_.find(*it) != std::string::npos) {
              multi_support = true;
            }
          }
          // The operator to be marked true, it should be supported by atleast CPU device specified with AUTO
          if (device_id_.find("AUTO") == 0) {
            if (std::string(*it).find("CPU") == std::string::npos) {
              auto_support = false;
            } else if ((*it == "All") || (device_id_.find(*it) != std::string::npos)) {
              auto_support = true;
            }
          }
          // if device supported is all then we support it
          if (*it == "All") {
            return true;
          }
          // check for device supported
          if (status == false) {
            if (device_id_.find(*it) != std::string::npos) {
              return true;
            }
          }
          it++;
        }
      }
    }
  }
  if (device_id_.find("AUTO") == 0 && auto_support == true) {
    return true;
  }
  if (device_id_.find("MULTI") == 0 && multi_support == true) {
    return true;
  }
  return false;
}

bool DataOps::type_is_supported(ONNXTensorElementDataType dtype, bool is_initializer) {
  if (is_initializer) {
    for (auto const& var : supported_types_initializer_) {
      if ((var.first <= version_id_) &&
          (var.second == dtype)) {
        return true;
      }
    }

#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Initializer Data Type is not supported" << std::endl;
    }
#endif
    return false;
  }
  if (device_id_.find("HETERO") != std::string::npos ||
      device_id_.find("MULTI") != std::string::npos || device_id_.find("AUTO") != std::string::npos) {
    for (auto const& var : supported_types_npu_) {
      if ((var.first <= version_id_) &&
          (var.second == dtype)) {
        return true;
      }
    }

#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "I/O data type is not supported" << std::endl;
    }
#endif
    return false;

  } else if (device_id_ == "CPU") {
    for (auto const& var : supported_types_cpu_) {
      if ((var.first <= version_id_) &&
          (var.second == dtype)) {
        return true;
      }
    }
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "I/O data type is not supported" << std::endl;
    }
#endif
    return false;

  } else if (device_id_ == "GPU") {
    for (auto const& var : supported_types_gpu_) {
      if ((var.first <= version_id_) &&
          (var.second == dtype)) {
        return true;
      }
    }
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "I/O data type is not supported" << std::endl;
    }
#endif
    return false;
  }
  return true;
}

bool DataOps::unsupported_op_mode(const OrtNode* node) {
  bool result = false;
  const char* optype = nullptr;
  graph_api_->OrtNode_GetOpType(node, &optype);
//  const char** initializers = nullptr;
//  size_t initializers_count = 0;
//  graph_api_->OrtGraph_GetAllInitializers(graph_viewer_, &initializers, &initializers_count);

  auto iter = op_list_.equal_range(std::string(optype));
  for (auto it = iter.first; it != iter.second; ++it) {
    auto ob = it->second;
    if (std::find(ob.ver.begin(), ob.ver.end(), version_id_) != ob.ver.end()) {
      return ob.func(node);
    }
  }
  return result;
}

bool DataOps::dimension_unsupported(const OrtNode* node) {
  const char* input0_name = nullptr;
  graph_api_->OrtNode_GetIthInputName(node, 0, &input0_name);
  OrtValueInfoRef* value_info = nullptr;
  graph_api_->OrtGraph_GetValueInfo(graph_viewer_, input0_name, &value_info);
  if (value_info->shape == nullptr) {
    graph_api_->OrtGraph_ReleaseValueInfo(value_info);
    return true;
  }
  size_t input_dims = value_info->shape_len;
  graph_api_->OrtGraph_ReleaseValueInfo(value_info);
  const char* optype = nullptr;
  graph_api_->OrtNode_GetOpType(node, &optype);
  if (!strstr(optype, "Pool") && input_dims != 4 && input_dims != 5) return false;

  if (!strcmp(optype, "ReduceSum")) {
    size_t key_count = 0;
    int axes_size = 0;
    graph_api_->OrtNode_GetAttributeKeyCount(node, "axes", &key_count);
    if (key_count > 0) graph_api_->OrtNode_GetAttributeIntSize(node, "axes", &axes_size);
    if (device_id_.find("GPU") != std::string::npos && axes_size == 0) return true;
    if (axes_size == 0) return false;
  }
  return true;
}

bool DataOps::node_is_supported(const NodeIndex node_idx) {
  const OrtNode* node = nullptr;
  graph_api_->OrtGraph_GetOrtNode(graph_viewer_, node_idx, &node);
  const char* optype = nullptr;
  graph_api_->OrtNode_GetOpType(node, &optype);

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Node " << optype << std::endl;
  }
#endif

  /*
  0. Check if node is in the unsupported list
  1. Check input and output data types are supported.
  2. Check if there is unsupported dimension in input and output shapes
  3. Check Op is supported
   3a. Check if Op is of known unsupported modes (edge cases). If yes return false right away.
   3b. If above is not true, check if the op is available in nGraph.
  */

  // Check 0
  if (!op_is_supported(optype, supported_op_mode)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is not in the supported ops list" << std::endl;
    }
#endif
    return false;
  }

  // Check 1
  bool are_types_supported = true;
  ForEachNodeDef(graph_api_, graph_viewer_, node, [&are_types_supported, this](const char* arg_name, const OrtValueInfoRef* node_arg, bool is_input) {
    bool is_initializer = false;
    if (is_input) {
      graph_api_->OrtGraph_IsConstantInitializer(graph_viewer_, arg_name, true, &is_initializer);
    }

    bool is_supported = type_is_supported(node_arg->data_type, is_initializer);
    are_types_supported &= is_supported;
  });

  if (!are_types_supported) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "DType is not supported" << std::endl;
    }
#endif
    return false;
  }

  // Check 2

  bool has_unsupported_dimension = false;
  ForEachNodeDef(graph_api_, graph_viewer_, node, [&has_unsupported_dimension, this, &optype, &node](const char* arg_name, const OrtValueInfoRef* node_arg, bool is_input) {
    if (is_input) {
      bool is_constant_initializer = false;
      graph_api_->OrtGraph_IsConstantInitializer(graph_viewer_, arg_name, true, &is_constant_initializer);
      if (is_constant_initializer)
        return;
    }

    if (node_arg->shape_len == 0) {
      if (op_is_supported(optype, no_dimension_supported_)) {
        return;
      }
      if (npu_qdq_optimizer_enabled_) {
        // Pad Op with DQ inputs will be optimized out in the qdq optimization pass, so mark those no dim Pad ops
        // supported here
        if (optype == "Pad") {
          size_t num_inputs = 0;
          graph_api_->OrtNode_GetNumInputs(node, &num_inputs);
          for (int i = 0; i < num_inputs; i++) {
            const char* input_name = nullptr;
            graph_api_->OrtNode_GetIthInputName(node, i, &input_name);
            const OrtNode* DQ = nullptr;
            graph_api_->OrtGraph_GetNodeProducingOutput(graph_viewer_, input_name, &DQ);
            const char* dq_optype = nullptr;
            graph_api_->OrtNode_GetOpType(DQ, &dq_optype);
            if (!strcmp(dq_optype, "DequantizeLinear")) return;
          }
        }
      }
      has_unsupported_dimension = true;
      return;
    }
    // Zero dimension check
    for (int i = 0; i < node_arg->shape_len; i++) {
      if (node_arg->shape[i] == 0) {
        if (((device_id_.find("CPU") != std::string::npos) || (device_id_.find("GPU") != std::string::npos)) &&
            (strcmp(optype, "Expand") == 0 || strcmp(optype, "Equal")  == 0 ||
             strcmp(optype, "Slice") == 0 || strcmp(optype, "Concat") == 0 ||
             strcmp(optype, "Shape") == 0)) {
          return;
        }
        has_unsupported_dimension = true;
        return;
      }
    }
  });
  if (has_unsupported_dimension) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Dimension check failed" << std::endl;
    }
#endif

    return false;
  }

  // Check 3a
  const char* domain = nullptr;
  graph_api_->OrtNode_GetDomain(node, &domain);
  if (strcmp(domain, kOnnxDomain) == 0 && unsupported_op_mode(node)) {
    if (optype == "GatherElements") {
      return true;
    }
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Failed in unsupported op mode" << std::endl;
    }
#endif
    return false;
  }

  return true;
}

std::vector<NodeIndex> DataOps::GetUnsupportedNodeIndices(std::unordered_set<std::string>& ng_required_initializers) {
  std::vector<NodeIndex> unsupported_nodes_idx;

  size_t num_nodes = 0;
  const size_t* nodes_topo_order = nullptr;
  graph_api_->OrtGraph_GetNodesIndexInTopologicalOrder(graph_viewer_, 0, &nodes_topo_order, &num_nodes);
  for (int i = 0; i < num_nodes; i++) {
    size_t node_idx = nodes_topo_order[i];
    if (node_is_supported(node_idx)) {
      // Collect inputs that are initializers
      const OrtNode* node = nullptr;
      graph_api_->OrtGraph_GetOrtNode(graph_viewer_, node_idx, &node);
      ForEachNodeDef(graph_api_, graph_viewer_, node, [&ng_required_initializers, this](const char* name, const OrtValueInfoRef* value_info, bool is_input) {
        if (is_input) {
          const char** initializer_names = nullptr;
          size_t initializers_count = 0;
          graph_api_->OrtGraph_GetAllInitializers(graph_viewer_, &initializer_names, &initializers_count);
          for (int j = 0; j < initializers_count; j++) {
            if (!strcmp(initializer_names[j], name)) ng_required_initializers.insert(std::string(name));
          }
          graph_api_->ReleaseCharArray(initializer_names);
        }
      });
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }
  return unsupported_nodes_idx;
}

bool DataOps::IsOpSupportedOnlyInModel(std::string name) {
  return ops_supported_only_in_model.find(name) != ops_supported_only_in_model.end();
}

bool DataOps::SpecialConditionForClusterSizeOne(std::unordered_set<std::string>& ng_required_initializers,
                                                const OrtNode* node) {
  const char* optype = nullptr;
  graph_api_->OrtNode_GetOpType(node, &optype);
  if (!strcmp(optype, "Reshape")) {
    const char* input_name = nullptr;
    graph_api_->OrtNode_GetIthInputName(node, 1, &input_name);
    if (ng_required_initializers.find(std::string(input_name)) == ng_required_initializers.end()) {
      return true;
    }
  } else if (!strcmp(optype, "Expand")) {
    // nGraph only supports constant shape input values
    const char* output_name = nullptr;
    graph_api_->OrtNode_GetIthOutputName(node, 0, &output_name);
    OrtValueInfoRef* value_info = nullptr;
    graph_api_->OrtGraph_GetValueInfo(graph_viewer_, output_name, &value_info);
    if (value_info->data_type != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      graph_api_->OrtGraph_ReleaseValueInfo(value_info);
      return true;
    }
    graph_api_->OrtGraph_ReleaseValueInfo(value_info);
  } else if (!strcmp(optype, "RoiAlign")) {
    const char* input0_name = nullptr, *input1_name = nullptr, *input2_name = nullptr, *output_name = nullptr;
    graph_api_->OrtNode_GetIthInputName(node, 0, &input0_name);
    graph_api_->OrtNode_GetIthInputName(node, 1, &input1_name);
    graph_api_->OrtNode_GetIthInputName(node, 2, &input2_name);
    graph_api_->OrtNode_GetIthOutputName(node, 0, &output_name);
    OrtValueInfoRef* input0_info = nullptr, *input1_info = nullptr, *input2_info = nullptr, *output_info = nullptr;
    graph_api_->OrtGraph_GetValueInfo(graph_viewer_, input0_name, &input0_info);
    graph_api_->OrtGraph_GetValueInfo(graph_viewer_, input1_name, &input1_info);
    graph_api_->OrtGraph_GetValueInfo(graph_viewer_, input2_name, &input2_info);
    graph_api_->OrtGraph_GetValueInfo(graph_viewer_, output_name, &output_info);

    if (input0_info->data_type != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
        input1_info->data_type != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
        input2_info->data_type != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info->data_type != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
      ) {
        graph_api_->OrtGraph_ReleaseValueInfo(input0_info);
        graph_api_->OrtGraph_ReleaseValueInfo(input1_info);
        graph_api_->OrtGraph_ReleaseValueInfo(input2_info);
        graph_api_->OrtGraph_ReleaseValueInfo(output_info);
        return true;
      }
      graph_api_->OrtGraph_ReleaseValueInfo(input0_info);
      graph_api_->OrtGraph_ReleaseValueInfo(input1_info);
      graph_api_->OrtGraph_ReleaseValueInfo(input2_info);
      graph_api_->OrtGraph_ReleaseValueInfo(output_info);
  }
  return false;
}

bool DataOps::DoNotOmitSubGraph(const std::string& name) {
  return op_is_supported(name, subgraph_supported_);
}

bool DataOps::InsertNode(const std::string& optype) {
  if (optype == "TopK" || optype == "NonZero") {
    return true;
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime
