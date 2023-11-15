// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include <unordered_set>
#include "../backend_utils.h"
#include "../backend_manager.h"
#include <string>
#include <vector>
#include "data_ops.h"
#include "capabilities.h"
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
    "LayerNormalization"};

std::vector<SupportedOp> supported_op_mode = {
    {"Abs", V_2020_4, {"CPU", "GPU"}},
    {"Abs", V_2023_0, {"VPUX"}},
    {"Acos", V_2020_4, {"CPU"}},
    {"Acos", V_2022_1, {"GPU"}},
    {"Acosh", V_2020_4, {"CPU"}},
    {"Acosh", V_2022_1, {"GPU"}},
    {"Add", V_2020_4, {"CPU", "GPU"}},
    {"Add", V_2023_0, {"VPUX"}},
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
    {"AveragePool", V_2023_0, {"VPUX"}},
    {"BatchNormalization", V_2020_4, {"CPU", "GPU"}},
    {"BatchNormalization", V_2023_0, {"VPUX"}},
    {"BitShift", V_2022_1, {"CPU"}},
    {"Cast", V_2020_4, {"CPU", "GPU"}},
    {"Cast", V_2023_0, {"VPUX"}},
    {"Ceil", V_2020_4, {"GPU"}},
    {"Ceil", V_2021_4, {"CPU"}},
    {"Celu", V_2022_1, {"CPU", "GPU"}},
    {"Clip", V_2020_4, {"CPU", "GPU"}},
    {"Clip", V_2023_0, {"VPUX"}},
    {"Concat", V_2020_4, {"CPU", "GPU"}},
    {"Concat", V_2023_0, {"VPUX"}},
    {"Constant", V_2020_4, {"CPU", "GPU"}},
    {"Constant", V_2023_0, {"VPUX"}},
    {"ConstantOfShape", V_2020_4, {"CPU", "GPU"}},
    {"ConstantOfShape", V_2023_0, {"VPUX"}},  // Gets mapped to broadcast op in the plugin.
    {"Conv", V_2020_4, {"CPU", "GPU"}},
    {"Conv", V_2023_0, {"VPUX"}},
    {"ConvInteger", V_2022_1, {"CPU", "GPU"}},
    {"ConvTranspose", V_2020_4, {"CPU", "GPU"}},
    {"Cos", V_2020_4, {"CPU"}},
    {"Cos", V_2022_1, {"GPU"}},
    {"Cos", V_2023_0, {"VPUX"}},
    {"Cosh", V_2020_4, {"CPU"}},
    {"Cosh", V_2022_1, {"GPU"}},
    {"CumSum", V_2022_1, {"CPU", "GPU"}},
    {"CumSum", V_2023_0, {"VPUX"}},
    {"DepthToSpace", V_2020_4, {"CPU", "GPU"}},
    {"DepthToSpace", V_2023_0, {"VPUX"}},
    {"DequantizeLinear", V_2021_4, {"CPU", "GPU"}},
    {"DequantizeLinear", V_2023_0, {"VPUX"}},
    {"Div", V_2020_4, {"CPU", "GPU"}},
    {"Div", V_2023_0, {"VPUX"}},
    {"Dropout", V_2020_4, {"CPU", "GPU"}},
    {"Dropout", V_2023_0, {"VPUX"}},
    {"Elu", V_2020_4, {"CPU", "GPU"}},
    {"Elu", V_2023_0, {"VPUX"}},
    // {"Einsum", V_2023_0, {"CPU", "GPU"}},
    {"Equal", V_2020_4, {"CPU", "GPU"}},
    {"Equal", V_2023_0, {"VPUX"}},  // Added for whisper decoder model.
    {"Erf", V_2020_4, {"CPU", "GPU"}},
    {"Erf", V_2023_0, {"VPUX"}},
    {"Exp", V_2020_4, {"CPU", "GPU"}},
    {"Exp", V_2023_0, {"VPUX"}},
    {"Expand", V_2022_1, {"CPU", "GPU"}},
    {"Expand", V_2023_0, {"VPUX"}},  // Gets mapped to broadcast op and multiply op in the plugin.
    {"EyeLike", V_2022_1, {"CPU"}},
    {"EyeLike", V_2023_0, {"VPUX"}},  // NoOP
    {"Flatten", V_2020_4, {"CPU", "GPU"}},
    {"Flatten", V_2023_0, {"VPUX"}},
    {"Floor", V_2020_4, {"CPU", "GPU"}},
    {"Gather", V_2020_4, {"CPU", "GPU"}},
    {"Gather", V_2023_0, {"VPUX"}},
    {"GatherElements", V_2022_2, {"CPU", "GPU"}},
    {"GatherND", V_2021_4, {"CPU", "GPU"}},
    {"Gemm", V_2020_4, {"CPU", "GPU"}},
    {"Gemm", V_2023_0, {"VPUX"}},
    {"GlobalAveragePool", V_2020_4, {"CPU", "GPU"}},
    {"GlobalAveragePool", V_2023_0, {"VPUX"}},
    {"GlobalLpPool", V_2020_4, {"CPU", "GPU"}},
    {"GlobalMaxPool", V_2022_1, {"CPU", "GPU"}},
    {"Greater", V_2020_4, {"CPU", "GPU"}},
    {"Greater", V_2023_0, {"VPUX"}},
    {"GreaterOrEqual", V_2022_1, {"CPU", "GPU"}},
    {"GreaterOrEqual", V_2023_0, {"VPUX"}},
    {"GridSample", V_2022_3, {"CPU"}},
    {"GridSample", V_2023_0, {"GPU"}},
    {"Identity", V_2020_4, {"CPU", "GPU"}},
    {"Identity", V_2023_0, {"VPUX"}},  // NoOP
    {"If", V_2022_3, {"CPU", "GPU"}},
    {"ImageScaler", V_2022_1, {"CPU", "GPU"}},
    {"ImageScaler", V_2023_0, {"VPUX"}},
    {"InstanceNormalization", V_2020_4, {"CPU", "GPU"}},
    {"InstanceNormalization", V_2023_0, {"VPUX"}},
    {"HardSigmoid", V_2020_4, {"CPU", "GPU"}},
    {"HardMax", V_2022_1, {"CPU", "GPU"}},
    {"LeakyRelu", V_2020_4, {"CPU", "GPU"}},
    {"LeakyRelu", V_2023_0, {"VPUX"}},
    {"Less", V_2020_4, {"CPU", "GPU"}},
    {"Less", V_2023_0, {"VPUX"}},  // Added for whisper decoder model.
    {"LessOrEqual", V_2022_1, {"CPU", "GPU"}},
    {"LessOrEqual", V_2023_0, {"VPUX"}},
    {"Log", V_2020_4, {"CPU", "GPU"}},
    {"Log", V_2023_0, {"VPUX"}},
    {"LogSoftMax", V_2022_1, {"CPU", "GPU"}},
    {"Loop", V_2021_4, {"CPU", "GPU"}},
    {"LRN", V_2020_4, {"CPU", "GPU"}},
    {"LRN", V_2023_0, {"VPUX"}},
    {"LSTM", V_2020_4, {"CPU", "GPU"}},
    {"MatMul", V_2020_4, {"CPU", "GPU"}},
    {"MatMul", V_2023_0, {"VPUX"}},
    {"MatMulInteger", V_2022_1, {"CPU"}},
    {"Max", V_2020_4, {"CPU", "GPU"}},
    {"Max", V_2023_0, {"VPUX"}},
    {"MaxPool", V_2020_4, {"CPU", "GPU"}},
    {"MaxPool", V_2023_0, {"VPUX"}},
    {"Mean", V_2020_4, {"CPU", "GPU"}},
    {"Mean", V_2023_0, {"VPUX"}},
    {"MeanVarianceNormalization", V_2022_1, {"CPU", "GPU"}},
    {"Min", V_2020_4, {"CPU", "GPU"}},
    {"Min", V_2023_0, {"VPUX"}},
    {"Mod", V_2022_1, {"CPU", "GPU"}},
    {"Mul", V_2020_4, {"CPU", "GPU"}},
    {"Mul", V_2023_0, {"VPUX"}},
    {"Neg", V_2020_4, {"CPU", "GPU"}},
    {"Neg", V_2023_0, {"VPUX"}},
    {"NonMaxSuppression", V_2021_1, {"CPU", "GPU"}},
    {"NonZero", V_2021_1, {"CPU"}},
    {"NonZero", V_2023_0, {"GPU"}},
    {"Not", V_2021_1, {"CPU", "GPU"}},
    {"Not", V_2020_4, {"CPU", "GPU"}},
    {"OneHot", V_2020_4, {"CPU", "GPU"}},
    {"Or", V_2022_1, {"CPU", "GPU"}},
    {"Pad", V_2020_4, {"CPU", "GPU"}},
    {"Pad", V_2023_0, {"VPUX"}},
    {"Pow", V_2020_4, {"CPU", "GPU"}},
    {"Pow", V_2023_0, {"VPUX"}},
    {"PRelu", V_2020_4, {"CPU", "GPU"}},
    {"PRelu", V_2023_0, {"VPUX"}},
    {"QLinearMatMul", V_2022_3, {"CPU"}},
    {"QuantizeLinear", V_2021_4, {"CPU", "GPU"}},
    {"QuantizeLinear", V_2023_0, {"VPUX"}},
    {"RandomNormalLike", V_2023_0, {"CPU", "GPU"}},
    {"RandomNormal", V_2023_0, {"CPU", "GPU"}},
    {"Range", V_2022_1, {"CPU", "GPU"}},
    {"Range", V_2023_0, {"VPUX"}},
    {"Reciprocal", V_2020_4, {"CPU", "GPU"}},
    {"Reciprocal", V_2023_0, {"VPUX"}},
    {"ReduceL1", V_2022_1, {"CPU", "GPU"}},
    {"ReduceL2", V_2022_1, {"CPU", "GPU"}},
    {"ReduceLogSum", V_2020_4, {"CPU"}},
    {"ReduceLogSum", V_2022_1, {"CPU", "GPU"}},
    {"ReduceLogSumExp", V_2022_1, {"CPU", "GPU"}},
    {"ReduceMax", V_2020_4, {"CPU", "GPU"}},
    {"ReduceMean", V_2020_4, {"CPU", "GPU"}},
    {"ReduceMean", V_2023_0, {"VPUX"}},
    {"ReduceMin", V_2020_4, {"CPU", "GPU"}},
    {"ReduceProd", V_2020_4, {"CPU"}},
    {"ReduceProd", V_2022_1, {"GPU"}},
    {"ReduceSum", V_2020_4, {"CPU", "GPU"}},
    {"ReduceSumSquare", V_2020_4, {"CPU"}},
    {"ReduceSumSquare", V_2022_1, {"CPU", "GPU"}},
    {"Relu", V_2020_4, {"CPU", "GPU"}},
    {"Relu", V_2023_0, {"VPUX"}},
    {"Resize", V_2020_4, {"CPU"}},
    {"Resize", V_2022_1, {"GPU"}},
    {"Reshape", V_2020_4, {"CPU", "GPU"}},
    {"Reshape", V_2023_0, {"VPUX"}},
    {"ReverseSequence", V_2022_1, {"CPU", "GPU"}},
    {"RoiAlign", V_2021_1, {"CPU", "GPU"}},
    {"Round", V_2021_4, {"CPU", "GPU"}},
    {"Scatter", V_2022_1, {"CPU", "GPU"}},
    {"ScatterElements", V_2022_1, {"CPU", "GPU"}},
    {"ScatterND", V_2022_1, {"CPU", "GPU"}},
    {"Selu", V_2020_4, {"CPU", "GPU"}},
    {"Shape", V_2020_4, {"CPU", "GPU"}},
    {"Shape", V_2023_0, {"VPUX"}},
    {"Shrink", V_2022_1, {"CPU", "GPU"}},
    {"Shrink", V_2023_0, {"VPUX"}},
    {"Sigmoid", V_2020_4, {"CPU", "GPU"}},
    {"Sigmoid", V_2023_0, {"VPUX"}},
    {"Sign", V_2020_4, {"CPU"}},
    {"Sign", V_2022_1, {"GPU"}},
    {"Sign", V_2023_0, {"VPUX"}},
    {"Sin", V_2022_1, {"CPU", "GPU"}},
    {"Sin", V_2023_0, {"VPUX"}},
    {"Sinh", V_2020_4, {"CPU"}},
    {"Size", V_2022_1, {"CPU", "GPU"}},
    {"Slice", V_2020_4, {"CPU", "GPU"}},
    {"Slice", V_2023_0, {"VPUX"}},
    {"Softmax", V_2020_4, {"CPU", "GPU"}},
    {"Softmax", V_2023_0, {"VPUX"}},
    {"Softplus", V_2022_1, {"CPU", "GPU"}},
    {"Softplus", V_2023_0, {"VPUX"}},
    {"Softsign", V_2022_1, {"CPU", "GPU"}},
    {"SpaceToDepth", V_2020_4, {"CPU", "GPU"}},
    {"SpaceToDepth", V_2023_0, {"VPUX"}},
    {"Split", V_2020_4, {"CPU", "GPU"}},
    {"Split", V_2023_0, {"VPUX"}},
    {"Sqrt", V_2020_4, {"CPU", "GPU"}},
    {"Sqrt", V_2023_0, {"VPUX"}},
    {"Squeeze", V_2020_4, {"CPU", "GPU"}},
    {"Squeeze", V_2023_0, {"VPUX"}},
    {"Softsign", V_2020_4, {"CPU"}},
    {"Sub", V_2020_4, {"CPU", "GPU"}},
    {"Sub", V_2023_0, {"VPUX"}},
    {"Sum", V_2020_4, {"CPU", "GPU"}},
    {"Sum", V_2023_0, {"VPUX"}},
    {"Tan", V_2020_4, {"CPU", "GPU"}},
    {"Tanh", V_2020_4, {"CPU", "GPU"}},
    {"Tanh", V_2023_0, {"VPUX"}},
    {"ThresholdedRelu", V_2022_1, {"CPU", "GPU"}},
    {"ThresholdedRelu", V_2023_0, {"VPUX"}},
    {"Tile", V_2021_3, {"CPU", "GPU"}},
    {"Tile", V_2023_0, {"VPUX"}},
    {"Transpose", V_2020_4, {"CPU", "GPU"}},
    {"Transpose", V_2023_0, {"VPUX"}},
    {"Trilu", V_2023_0, {"CPU", "GPU"}},
    {"TopK", V_2020_4, {"CPU", "GPU"}},
    {"TopK", V_2023_0, {"VPUX"}},
    {"Unsqueeze", V_2020_4, {"CPU", "GPU"}},
    {"Unsqueeze", V_2023_0, {"VPUX"}},
    {"Upsample", V_2021_1, {"CPU"}},
    {"Upsample", V_2021_4, {"GPU"}},
    {"Upsample", V_2023_0, {"VPUX"}},
    {"Where", V_2022_1, {"CPU", "GPU"}},
    {"Where", V_2023_0, {"VPUX"}},  // Added for whisper decoder model.
    {"Xor", V_2022_1, {"CPU", "GPU"}},
};

void DataOps::populate_types_supported() {
  supported_types_initializer_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::BOOL)));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::FLOAT)));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT32)));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT64)));
  supported_types_initializer_.insert(std::make_pair(V_2021_1, static_cast<int>(onnxruntime::DataType::FLOAT16)));
  supported_types_initializer_.insert(std::make_pair(V_2021_4, static_cast<int>(onnxruntime::DataType::INT8)));
  supported_types_initializer_.insert(std::make_pair(V_2021_4, static_cast<int>(onnxruntime::DataType::UINT8)));

  supported_types_vpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::BOOL)));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::FLOAT)));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::UINT8)));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT8)));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT16)));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT32)));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT64)));
  supported_types_vpu_.insert(std::make_pair(V_2021_1, static_cast<int>(onnxruntime::DataType::FLOAT16)));

  supported_types_cpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::BOOL)));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::FLOAT)));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT32)));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT16)));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT8)));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::UINT8)));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT64)));
  supported_types_cpu_.insert(std::make_pair(V_2022_2, static_cast<int>(onnxruntime::DataType::FLOAT16)));

  supported_types_gpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::FLOAT)));
  supported_types_gpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT32)));
  supported_types_gpu_.insert(std::make_pair(V_2020_4, static_cast<int>(onnxruntime::DataType::INT64)));
  supported_types_gpu_.insert(std::make_pair(V_2021_1, static_cast<int>(onnxruntime::DataType::FLOAT16)));
  supported_types_gpu_.insert(std::make_pair(V_2021_4, static_cast<int>(onnxruntime::DataType::INT8)));
  supported_types_gpu_.insert(std::make_pair(V_2021_4, static_cast<int>(onnxruntime::DataType::UINT8)));
  supported_types_gpu_.insert(std::make_pair(V_2022_1, static_cast<int>(onnxruntime::DataType::BOOL)));
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
  no_dimension_supported_.push_back({"Floor", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Gather", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Greater", V_2023_0, {"VPUX"}});
  no_dimension_supported_.push_back({"Less", V_2022_1, {"CPU"}});
  no_dimension_supported_.push_back({"Loop", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Max", V_2023_0, {"VPUX"}});
  no_dimension_supported_.push_back({"Min", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Mul", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"QuantizeLinear", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"Range", V_2021_2, {"All"}});
  no_dimension_supported_.push_back({"ReduceMax", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"ReduceMin", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"ReduceProd", V_2022_1, {"CPU", "GPU"}});
  no_dimension_supported_.push_back({"Reshape", V_2022_1, {"All"}});
  no_dimension_supported_.push_back({"Shape", V_2022_1, {"GPU"}});
  no_dimension_supported_.push_back({"Shape", V_2023_0, {"CPU"}});
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
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // Abs is not supproted with INT8 or INT32 as input data type on GPU
                               if (device_id_.find("GPU") != std::string::npos) {
                                 std::vector<std::string_view> inputs = node->Inputs();
                                 for (size_t i = 0; i < inputs.size(); i++) {
                                   onnxruntime::DataType type = graph_viewer_.GetValueInfoView(inputs[i])->DType();
                                   if (type == onnxruntime::DataType::INT8 || type == onnxruntime::DataType::INT32) return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Abs", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // tensor type does not support select last index
                               std::optional<int64_t> attribute = node->GetAttributeInt("select_last_index");
                               int64_t last_index_arg = 0;
                               if (attribute) last_index_arg = *attribute;
                               if (last_index_arg != 0) return true;
                               // tensor type supports float as input for argmax and argmin
                               if (graph_viewer_.GetValueInfoView(node->Inputs()[0])->DType() != onnxruntime::DataType::FLOAT) return true;
                               return false;
                             }};
    op_list_.insert({"ArgMax", obj});
    op_list_.insert({"ArgMin", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 // int64 data type is not supported on GPU
                                 const bool data_is_int64 = graph_viewer_.GetValueInfoView(node->Inputs()[0])->DType() != onnxruntime::DataType::INT64; // TODO: Review
                                 return data_is_int64;
                               }
                               return false;
                             }};
    op_list_.insert({"Clip", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 bool if_bias = false;
                                 std::optional<std::vector<int64_t>> conv_filter = node->GetAttributeInts("kernel_shape");
                                 if (conv_filter) {
                                   // check if the Input for the op has bias
                                  if (node->Inputs().size() > 2 && node->Inputs()[2] == "B") if_bias = true;
                                   // If the kernel size is 3D and the input doesnot have bias,
                                   // the op is rejected in case of GPU
                                  if ((*conv_filter).size() == 3 && !if_bias) return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Conv", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 // If the device is GPU, only 2D dilations with 1x1 pixel are supported
                                 std::optional<std::vector<int64_t>> dilation = node->GetAttributeInts("dilations");
                                 if (dilation) {
                                  if ((*dilation).size() == 2 && ((*dilation)[0] != 1) || ((*dilation)[1] != 1)) return true;
                                  // If 3D dilations, reject the op
                                  if ((*dilation).size() == 3)
                                     return true;
                                 }
                                 std::optional<int64_t> group_attr = node->GetAttributeInt("group");
                                 // group 4 is not supported
                                 if (group_attr && *group_attr == 4) return true;
                               }
                               return false;
                             }};
    op_list_.insert({"ConvTranspose", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos && node->OpType() == "If") {
                                 // Only Equal op is supported as input for IF op in GPU
                                 for (std::string_view input : node->Inputs()) {
                                   std::unique_ptr<interface::NodeViewRef> nit = graph_viewer_.GetNodeViewProducingOutput(input);
                                   if (nit->OpType() == "Equal") return false;
                                 }
                               }
                               return true;
                             }};
    op_list_.insert({"If", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // dilations attrs are not supported yet for Maxpool
                               if (node->GetAttributeInt("dilations") || node->GetAttributeString("dilations") || node->GetAttributeInts("dilations")) return true;
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"MaxPool", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 onnxruntime::DataType x_data_type = graph_viewer_.GetValueInfoView(node->Inputs()[0])->DType();
                                 onnxruntime::DataType y_data_type = graph_viewer_.GetValueInfoView(node->Inputs()[1])->DType();
                                 // currently both inputs with int32 are not supported and also both input datatypes should be same
                                 if (x_data_type == onnxruntime::DataType::INT32 && y_data_type == onnxruntime::DataType::INT32 || x_data_type != y_data_type) return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Mod", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               onnxruntime::DataType x_data_type = graph_viewer_.GetValueInfoView(node->Inputs()[0])->DType();
                               onnxruntime::DataType y_data_type = graph_viewer_.GetValueInfoView(node->Inputs()[1])->DType();
                               if (device_id_.find("GPU") != std::string::npos) {
                                 return x_data_type != y_data_type;
                               }
                               // currently both inputs with int32 or int64 datatype are not supported
                               if ((x_data_type == onnxruntime::DataType::INT32 && y_data_type == onnxruntime::DataType::INT32) ||
                                 (x_data_type == onnxruntime::DataType::INT64 && y_data_type == onnxruntime::DataType::INT64)) return true;
                               return false;
                             }};
    op_list_.insert({"Pow", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // Max op with one input is not supporting for GPU_FP16
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto prec_str = openvino_ep::BackendManager::GetGlobalContext().precision_str;
                                 if (prec_str == "FP16" && node->Inputs().size() == 1) return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Max", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // Min op with one input is not supporting for GPU_FP16
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto prec_str = openvino_ep::BackendManager::GetGlobalContext().precision_str;
                                 if (prec_str == "FP16" && node->Inputs().size() == 1) return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Min", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // Sum op with one input is not supporting for GPU_FP16
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto prec_str = openvino_ep::BackendManager::GetGlobalContext().precision_str;
                                 if (prec_str == "FP16" && node->Inputs().size() == 1) return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Sum", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 // PRelu slope has to be an initializer or needs to come from a constant node
                                 if (graph_viewer_.HasInitializerName(node->Inputs()[1])) return false;
                                 else {
                                   for (std::string_view input : node->Inputs()) {
                                    std::unique_ptr<interface::NodeViewRef> input_node = graph_viewer_.GetNodeViewProducingOutput(input);
                                    int count = 0;
                                    for (std::string_view input2 : input_node->Inputs()) {
                                      if (!graph_viewer_.HasInitializerName(input2)) count++;
                                    }
                                    if (count == 0) return false;
                                   }
                                 }
                               }
                               return true;
                             }};
    op_list_.insert({"PRelu", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3, V_2023_0},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               std::unique_ptr<interface::ValueInfoViewRef> input_arg = graph_viewer_.GetValueInfoView(node->Inputs()[1]);
                               std::optional<std::vector<int64_t>> shape = input_arg->Shape();
                               // Reshape op with empty dim is Rejected for Myriad
                               //[TODO] Is this condition required anymore with Myriad removed?
                               if (shape) {
                                 for (int64_t& dim : *shape) {
                                   if (dim == 0) return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Reshape", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               std::optional<std::string> attribute = node->GetAttributeString("mode");
                               if (attribute && *attribute == "linear" && node->Inputs().size() == 4) return true;
                               return false;
                             }};
    op_list_.insert({"Resize", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 // INT32 dataype is not supported as input
                                 for (std::string_view input : node->Inputs()) {
                                   if (graph_viewer_.GetValueInfoView(input)->DType() == onnxruntime::DataType::INT32) return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"ReduceLogSumExp", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 // If the output of ScatterND op is BOOL, it is rejected for GPU.
                                 if (graph_viewer_.GetValueInfoView(node->Outputs()[0])->DType() == onnxruntime::DataType::BOOL) return true;
                               }
                               return false;
                             }};
    op_list_.insert({"ScatterND", obj});
    op_list_.insert({"ScatterElements", obj});
    op_list_.insert({"Scatter", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // If the Input of Shrink op is UINT8, it is rejected (Due to output mismatch)
                               for (std::string_view input : node->Inputs()) {
                                 if (graph_viewer_.GetValueInfoView(input)->DType() == onnxruntime::DataType::UINT8) return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Shrink", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // start, end, axes need to be a initializer
                               bool cond_for_slice = false;
                               std::string_view data_arg = node->Inputs()[0];
                               auto graph_inputs = graph_viewer_.GetInputs();

                               auto it = find(graph_inputs.begin(), graph_inputs.end(), data_arg);
                               if (it != graph_inputs.end()) {
                                 if (node->Inputs().size() > 1) {
                                   std::string_view start_arg = node->Inputs()[1];
                                   std::string_view end_arg = node->Inputs()[2];
                                   cond_for_slice |= !graph_viewer_.HasInitializerName(start_arg);
                                   cond_for_slice |= !graph_viewer_.HasInitializerName(end_arg);
                                 }
                                 if (node->Inputs().size() > 3) {
                                   std::string_view axes_arg = node->Inputs()[3];
                                   cond_for_slice |= !graph_viewer_.HasInitializerName(axes_arg);
                                 }
                               }

                               return cond_for_slice;
                             }};
    op_list_.insert({"Slice", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 if (node->Inputs().size() > 1 && graph_viewer_.GetValueInfoView(node->Inputs()[0])->DType() == onnxruntime::DataType::FLOAT) return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Squeeze", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3, V_2023_0},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // If the operator is unsqueeze
                               // If axes is an input, then we cannot produce a static graph. Conversion fails in convert_function_to_cnn_network.
                               for (std::string_view input : node->Inputs()) {
                                 if (input == "axes") return true;
                               }
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"Unsqueeze", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3, V_2023_0},
                             [this](const onnxruntime::interface::NodeViewRef* node) {
                               // check for attributes
                               std::optional<std::vector<float>> upsample_attr = node->GetAttributeFloats("scales");
                               if (upsample_attr) {
                                 if ((*upsample_attr).size() > 2 && ((*upsample_attr)[0] != 1.f || (*upsample_attr)[1] != 1.f)) return true;
                               }

                               // check for input dimensions
                               std::unique_ptr<interface::ValueInfoViewRef> x_arg = graph_viewer_.GetValueInfoView(node->Inputs()[0]);
                               std::optional<std::vector<int64_t>> shape = x_arg->Shape();
                               if (shape && ((*shape).size() == 1 || (*shape).size() == 4)) return true;
                               // x_arg supports only float, int8 and float16 type
                               if (x_arg->DType() == onnxruntime::DataType::FLOAT || x_arg->DType() == onnxruntime::DataType::INT8 || x_arg->DType() == onnxruntime::DataType::FLOAT16) return false;
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

bool DataOps::type_is_supported(const onnxruntime::interface::ValueInfoViewRef* node_arg, bool is_initializer) {
  if (is_initializer) {
    int dtype = static_cast<int>(node_arg->DType());
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
  } else {
    int dtype = static_cast<int>(node_arg->DType());

    if (device_id_.find("VPUX") != std::string::npos || device_id_.find("HETERO") != std::string::npos ||
        device_id_.find("MULTI") != std::string::npos || device_id_.find("AUTO") != std::string::npos) {
      for (auto const& var : supported_types_vpu_) {
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
}

bool DataOps::unsupported_op_mode(const onnxruntime::interface::NodeViewRef* node) {
  bool result = false;
  std::string_view optype = node->OpType();

  auto iter = op_list_.equal_range(std::string(optype));
  for (auto it = iter.first; it != iter.second; ++it) {
    auto ob = it->second;
    if (std::find(ob.ver.begin(), ob.ver.end(), version_id_) != ob.ver.end()) {
      return ob.func(node);
    }
  }
  return result;
}

bool DataOps::dimension_unsupported(const onnxruntime::interface::NodeViewRef* node) {
  std::unique_ptr<interface::ValueInfoViewRef> input = graph_viewer_.GetValueInfoView(node->Inputs()[0]);
  size_t input_dims = 0;
  if (!input->Shape()) return true;
  input_dims = (*(input->Shape())).size();
  if (node->OpType().find("Pool") != std::string::npos && input_dims != 4 && input_dims != 5) return false;

  if (node->OpType() == "ReduceSum") {
    std::optional<std::vector<int64_t>> attributes = node->GetAttributeInts("axes");
    int64_t axes_size = attributes ? (*attributes).size() : 0;
    if (axes_size == 0) return device_id_.find("GPU") != std::string::npos;
  }
  return true;
}

bool DataOps::node_is_supported(const std::map<std::string, std::set<std::string>>& op_map,
                                const interface::NodeViewRef* node) {
  std::string_view optype = node->OpType();

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Node " << optype << std::endl;
  }
#endif

  std::string_view domain = node->Domain();

  /*
  0. Check if node is in the unsupported list
  1. Check input and output data types are supported.
  2. Check if there is unsupported dimension in input and output shapes
  3. Check Op is supported
   3a. Check if Op is of known unsupported modes (edge cases). If yes return false right away.
   3b. If above is not true, check if the op is available in nGraph.
  */

  // Check 0
  if (!op_is_supported(std::string(optype), supported_op_mode)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is not in the supported ops list" << std::endl;
    }
#endif
    return false;
  }

  // Check 1
  bool are_types_supported = true;

  node->ForEachDef([this, &are_types_supported](const interface::ValueInfoViewRef& node_arg, bool is_input) {
    bool is_initializer = false;
    if (is_input) {
      if (this->graph_viewer_.IsConstantInitializer(node_arg.Name(), true))
        is_initializer = true;
    }
    bool is_supported = type_is_supported(&node_arg, is_initializer);
    are_types_supported &= is_supported;
  }, false);

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
  node->ForEachDef([&has_unsupported_dimension, this, &optype](const interface::ValueInfoViewRef& node_arg, bool is_input) {
    if (is_input) {
      if (this->graph_viewer_.IsConstantInitializer(node_arg.Name(), true))
        return;
    }
    std::optional<std::vector<int64_t>> shape = node_arg.Shape();
    if (shape) {
      // Can't have no dimensions
      if ((*shape).size() == 0) {
        if (op_is_supported(std::string(optype), no_dimension_supported_)) {
          return;
        }
        if ((optype == "Identity") || (optype == "Sqrt")) {
          return;
        }
        has_unsupported_dimension = true;
        return;
      } else {
        // Zero dimension check
        for (int64_t& dim : *shape) {
          if (dim == 0) {
            if (((device_id_.find("CPU") != std::string::npos) || (device_id_.find("GPU") != std::string::npos)) &&
                ((optype == "Expand") || (optype == "Equal") ||
                 (optype == "Slice") || (optype == "Concat") ||
                 (optype == "Shape"))) {
              return;
            }
            has_unsupported_dimension = true;
            return;
          }
        }
      }
    }
  }, false);
  if (has_unsupported_dimension) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Dimension check failed" << std::endl;
    }
#endif

    return false;
  }

  // Check 3a
  if (domain == "" && unsupported_op_mode(node)) {
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

  // Check 3b
  const auto opset = op_map.find(std::string(domain));
  const auto op_fun = ops_supported_as_function.find(std::string(node->OpType()));
  if (opset == op_map.end()) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Failed in Unsupported onnx model domain" << std::endl;
    }
#endif
    return false;
  }
  if (opset->second.find(std::string(optype)) == opset->second.end() && op_fun == ops_supported_as_function.end()) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "The operator is not available in OpenVINO ngraph operators list nor the operator is a special ONNX function" << std::endl;
    }
#endif
    return false;
  }
  return true;
}

std::vector<interface::NodeViewRef*> DataOps::GetUnsupportedNodeIndices(std::unordered_set<std::string>& ng_required_initializers) {
  const auto ng_supported_ops = GetNgSupportedOps(GetOnnxOpSet(graph_viewer_));

  std::vector<interface::NodeViewRef*> unsupported_nodes;

  for (std::unique_ptr<interface::NodeViewRef>& node : graph_viewer_.NodeViews()) {
    if (node_is_supported(ng_supported_ops, node.get())) {
      // Collect inputs that are initializers
      node->ForEachDef([&ng_required_initializers, this](const interface::ValueInfoViewRef& node_arg, bool is_input) {
            if(is_input && this->graph_viewer_.HasInitializerName(node_arg.Name())) {
                ng_required_initializers.insert(std::string(node_arg.Name()));
              } }, true);
    } else {
      unsupported_nodes.push_back(node.get());
    }
  }
  return unsupported_nodes;
}

bool DataOps::IsOpSupportedOnlyInModel(std::string name) {
  return ops_supported_only_in_model.find(name) != ops_supported_only_in_model.end();
}

bool DataOps::SpecialConditionForClusterSizeOne(std::unordered_set<std::string>& ng_required_initializers, const onnxruntime::interface::NodeViewRef* node) {
  if (node->OpType() == "Reshape") {
    std::string_view shape_arg = node->Inputs()[1];
    if (ng_required_initializers.find(std::string(shape_arg)) == ng_required_initializers.end()) {
      return true;
    }
  } else if (node->OpType() == "Expand") {
    // nGraph only supports constant shape input values
    if (graph_viewer_.GetValueInfoView(node->Outputs()[0])->DType() != onnxruntime::DataType::FLOAT16) return true;
  } else if (node->OpType() == "RoiAlign") {
    onnxruntime::DataType input_0_data_type = graph_viewer_.GetValueInfoView(node->Inputs()[0])->DType();
    onnxruntime::DataType input_1_data_type = graph_viewer_.GetValueInfoView(node->Inputs()[1])->DType();
    onnxruntime::DataType input_2_data_type = graph_viewer_.GetValueInfoView(node->Inputs()[2])->DType();
    onnxruntime::DataType output_data_type = graph_viewer_.GetValueInfoView(node->Outputs()[0])->DType();

    if ((input_0_data_type != onnxruntime::DataType::FLOAT16) ||
        (input_1_data_type != onnxruntime::DataType::FLOAT16) ||
        (input_2_data_type != onnxruntime::DataType::FLOAT) ||
        (output_data_type != onnxruntime::DataType::FLOAT16))
      return true;
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
