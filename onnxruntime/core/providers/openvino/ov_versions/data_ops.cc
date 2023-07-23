// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include <unordered_set>
#include "core/providers/shared_library/provider_api.h"
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
#include <ngraph/ngraph.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>
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
  supported_types_initializer_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_initializer_.insert(std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));
  supported_types_initializer_.insert(std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_initializer_.insert(std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));

  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_vpu_.insert(std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));

  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_cpu_.insert(std::make_pair(V_2022_2, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));

  supported_types_gpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_gpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_gpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_gpu_.insert(std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));
  supported_types_gpu_.insert(std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_gpu_.insert(std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_gpu_.insert(std::make_pair(V_2022_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
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
                             [this](const Node* node, const InitializedTensorSet&) {
                               // Abs is not supproted with INT8 or INT32 as input data type on GPU
                               if (device_id_.find("GPU") != std::string::npos) {
                                 for (size_t i = 0; i < node->InputDefs().size(); i++) {
                                   if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8 ||
                                       node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32)
                                     return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Abs", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // tensor type does not support select last index
                               auto& attributes = node->GetAttributes();
                               auto last_index_arg = attributes.count("select_last_index") > 0 ? attributes.at("select_last_index").i() : 0;
                               if (last_index_arg != 0)
                                 return true;
                               // tensor type supports float as input for argmax and argmin
                               if (node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
                                 return true;
                               return false;
                             }};
    op_list_.insert({"ArgMax", obj});
    op_list_.insert({"ArgMin", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 // int64 data type is not supported on GPU
                                 const bool data_is_int64 = node->InputDefs()[0]->Type()->find("int64") != std::string::npos;
                                 return data_is_int64;
                               }
                               return false;
                             }};
    op_list_.insert({"Clip", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 bool if_bias = false;
                                 const auto& attributes = node->GetAttributes();
                                 auto conv_filter = attributes.find("kernel_shape");
                                 if (conv_filter != attributes.end()) {
                                   auto& ints = conv_filter->second().ints();
                                   // check if the Input for the op has bias
                                   if (node->InputDefs().size() > 2) {
                                     if (node->InputDefs()[2]->Name() == "B")
                                       if_bias = true;
                                   }
                                   // If the kernel size is 3D and the input doesnot have bias,
                                   // the op is rejected in case of GPU
                                   if (ints.size() == 3 && !if_bias)
                                     return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Conv", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 // If the device is GPU, only 2D dilations with 1x1 pixel are supported
                                 const auto& attributes = node->GetAttributes();
                                 auto dilation = attributes.find("dilations");
                                 if (dilation != attributes.end()) {
                                   auto& dilation_attr = attributes.at("dilations");
                                   auto int_size = dilation_attr.ints_size();
                                   if (int_size == 2) {
                                     if (dilation_attr.ints(0) != 1 || dilation_attr.ints(1) != 1) {
                                       return true;
                                     }
                                   }
                                   // If 3D dilations, reject the op
                                   if (int_size == 3)
                                     return true;
                                 }
                                 auto group_attr = attributes.find("group");
                                 // group 4 is not supported
                                 if (group_attr->second().i() == 4)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"ConvTranspose", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos && node->OpType() == "If") {
                                 // Only Equal op is supported as input for IF op in GPU
                                 for (auto nit = node->InputNodesBegin(); nit != node->InputNodesEnd(); ++nit) {
                                   if (nit->OpType() == "Equal") {
                                     return false;
                                   }
                                 }
                               }
                               return true;
                             }};
    op_list_.insert({"If", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               const auto& attributes = node->GetAttributes();
                               // dilations attrs are not supported yet for Maxpool
                               if (attributes.find("dilations") != attributes.end())
                                 return true;
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"MaxPool", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto x_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                 auto y_data_type = node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
                                 // currently both inputs with int32 are not supported and also both input datatypes should be same
                                 const bool A_is_int32 = node->InputDefs()[0]->Type()->find("int32") != std::string::npos;
                                 const bool B_is_int32 = node->InputDefs()[1]->Type()->find("int32") != std::string::npos;
                                 if ((A_is_int32 && B_is_int32) || (x_data_type != y_data_type))
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Mod", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto x_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                 auto y_data_type = node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
                                 return x_data_type != y_data_type;
                               }
                               // currently both inputs with int32 or int64 datatype are not supported
                               const bool A_is_int32 = node->InputDefs()[0]->Type()->find("int32") != std::string::npos;
                               const bool B_is_int32 = node->InputDefs()[1]->Type()->find("int32") != std::string::npos;
                               const bool A_is_int64 = node->InputDefs()[0]->Type()->find("int64") != std::string::npos;
                               const bool B_is_int64 = node->InputDefs()[1]->Type()->find("int64") != std::string::npos;
                               if ((A_is_int32 && B_is_int32) || (A_is_int64 && B_is_int64))
                                 return true;
                               return false;
                             }};
    op_list_.insert({"Pow", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // Max op with one input is not supporting for GPU_FP16
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto prec_str = openvino_ep::BackendManager::GetGlobalContext().precision_str;
                                 if (prec_str == "FP16") {
                                   if (node->InputDefs().size() == 1) {
                                     return true;
                                   }
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Max", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // Min op with one input is not supporting for GPU_FP16
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto prec_str = openvino_ep::BackendManager::GetGlobalContext().precision_str;
                                 if (prec_str == "FP16") {
                                   if (node->InputDefs().size() == 1) {
                                     return true;
                                   }
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Min", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // Sum op with one input is not supporting for GPU_FP16
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto prec_str = openvino_ep::BackendManager::GetGlobalContext().precision_str;
                                 if (prec_str == "FP16") {
                                   if (node->InputDefs().size() == 1) {
                                     return true;
                                   }
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Sum", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto slope = node->InputDefs()[1];
                                 // PRelu slope has to be an initializer or needs to come from a constant node
                                 if (initializers.count(slope->Name()))
                                   return false;
                                 else {
                                   for (auto input_node = node->InputNodesBegin(); input_node != node->InputNodesEnd(); ++input_node) {
                                     if (GetInputCount(this->graph_viewer_.GetNode((*input_node).Index()), initializers) == 0)
                                       return false;
                                   }
                                 }
                               }
                               return true;
                             }};
    op_list_.insert({"PRelu", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3, V_2023_0},
                             [this](const Node* node, const InitializedTensorSet&) {
                               const auto& input_arg = node->InputDefs()[1];
                               auto shape = input_arg->Shape();
                               // Reshape op with empty dim is Rejected for Myriad
                               //[TODO] Is this condition required anymore with Myriad removed?
                               if (shape != nullptr) {
                                 for (const auto& dim : input_arg->Shape()->dim()) {
                                   if (utils::HasDimValue(dim) && dim.dim_value() == 0)
                                     return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Reshape", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1},
                             [this](const Node* node, const InitializedTensorSet&) {
                               auto& attributes = node->GetAttributes();
                               if (attributes.count("mode") == 1 && attributes.at("mode").s() == "linear") {
                                 if (node->InputDefs().size() == 4) {
                                   return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Resize", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 // INT32 dataype is not supported as input
                                 for (size_t i = 0; i < node->InputDefs().size(); i++) {
                                   if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32)
                                     return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"ReduceLogSumExp", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                 // If the output of ScatterND op is BOOL, it is rejected for GPU.
                                 if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"ScatterND", obj});
    op_list_.insert({"ScatterElements", obj});
    op_list_.insert({"Scatter", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // If the Input of Shrink op is UINT8, it is rejected (Due to output mismatch)
                               for (size_t i = 0; i < node->InputDefs().size(); i++) {
                                 if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Shrink", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               // start, end, axes need to be a initializer
                               bool cond_for_slice = false;
                               const auto& data_arg = node->InputDefs()[0];
                               auto graph_inputs = graph_viewer_.GetInputs();

                               auto it = find(graph_inputs.begin(), graph_inputs.end(), data_arg);
                               if (it != graph_inputs.end()) {
                                 if (node->InputDefs().size() > 1) {
                                   const auto& start_arg = node->InputDefs()[1];
                                   const auto& end_arg = node->InputDefs()[2];
                                   cond_for_slice |= initializers.find(start_arg->Name()) == initializers.end();
                                   cond_for_slice |= initializers.find(end_arg->Name()) == initializers.end();
                                 }
                                 if (node->InputDefs().size() > 3) {
                                   const auto& axes_arg = node->InputDefs()[3];
                                   cond_for_slice |= initializers.find(axes_arg->Name()) == initializers.end();
                                 }
                               }

                               return cond_for_slice;
                             }};
    op_list_.insert({"Slice", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 if (node->InputDefs().size() > 1 &&
                                     (node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() ==
                                      ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)) {
                                   return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Squeeze", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3, V_2023_0},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // If the operator is unsqueeze
                               // If axes is an input, then we cannot produce a static graph. Conversion fails in convert_function_to_cnn_network.
                               for (size_t i = 0; i < node->InputDefs().size(); i++) {
                                 if (node->InputDefs()[i]->Name() == "axes") {
                                   return true;
                                 }
                               }
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"Unsqueeze", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2022_1, V_2022_2, V_2022_3, V_2023_0},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // check for attributes
                               auto& upsample_attr = node->GetAttributes();
                               if (upsample_attr.count("scales") > 0) {
                                 auto& upsample_arg = upsample_attr.at("scales");
                                 auto float_size = upsample_arg.floats_size();
                                 if (float_size > 2 && (upsample_arg.floats(0) != 1.f || upsample_arg.floats(1) != 1.f)) {
                                   return true;
                                 }
                               }

                               // check for input dimensions
                               const auto& x_arg = node->InputDefs()[0];
                               auto shape = x_arg->Shape();
                               if (shape != nullptr) {
                                 // input tensor rank cannot be of one dimension
                                 if (shape->dim_size() == 1 || shape->dim_size() == 4) {
                                   return true;
                                 }
                               }
                               // x_arg supports only float, int8 and float16 type
                               if ((x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) ||
                                   (x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) ||
                                   (x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)) {
                                 return false;
                               } else {
                                 return true;
                               }
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

bool DataOps::type_is_supported(const NodeArg* node_arg, bool is_initializer) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is not a proto " << std::endl;
    }
#endif
    return false;
  }

  if (is_initializer) {
    auto dtype = type_proto->tensor_type().elem_type();
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
    auto dtype = type_proto->tensor_type().elem_type();

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

bool DataOps::unsupported_op_mode(const Node* node) {
  bool result = false;
  const auto& optype = node->OpType();
  const auto& initializers = graph_viewer_.GetAllInitializedTensors();

  auto iter = op_list_.equal_range(optype);
  for (auto it = iter.first; it != iter.second; ++it) {
    auto ob = it->second;
    if (std::find(ob.ver.begin(), ob.ver.end(), version_id_) != ob.ver.end()) {
      return ob.func(node, initializers);
    }
  }
  return result;
}

bool DataOps::dimension_unsupported(const Node* node) {
  auto node_inputs = node->InputDefs();
  size_t input_dims = 0;
  if (node_inputs[0]->Shape() == nullptr) {
    return true;
  } else {
    input_dims = node_inputs[0]->Shape()->dim_size();
    if (node->OpType().find("Pool") != std::string::npos) {
      if (input_dims != 4 && input_dims != 5)
        return false;
    }
    /*
    if (node->OpType() == "Unsqueeze") {
      auto& attributes = node->GetAttributes();
      int64_t axes_size = attributes.count("axes") > 0 ? attributes.at("axes").ints().size() : 0;
      if (device_id_.find("GPU") != std::string::npos) {
        if (axes_size == 0)
          return true;
      }
      if (input_dims + axes_size > 5 || axes_size == 0) {
        return false;
      }
    }
    */

    if (node->OpType() == "ReduceSum") {
      auto& attributes = node->GetAttributes();
      int64_t axes_size = attributes.count("axes") > 0 ? attributes.at("axes").ints().size() : 0;
      if (device_id_.find("GPU") != std::string::npos) {
        if (axes_size == 0)
          return true;
      }
      if (axes_size == 0)
        return false;
    }
  }
  return true;
}

bool DataOps::node_is_supported(const std::map<std::string, std::set<std::string>>& op_map,
                                const NodeIndex node_idx) {
  const auto& node = graph_viewer_.GetNode(node_idx);
  const auto& optype = node->OpType();

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Node " << optype << std::endl;
  }
#endif

  const auto& domain = node->Domain();

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

  node->ForEachDef([this, &are_types_supported](const NodeArg& node_arg, bool is_input) {
    bool is_initializer = false;
    if (is_input) {
      if (this->graph_viewer_.IsConstantInitializer(node_arg.Name(), true))
        is_initializer = true;
    }
    bool is_supported = type_is_supported(&node_arg, is_initializer);
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
  node->ForEachDef([&has_unsupported_dimension, this, &optype](const NodeArg& node_arg, bool is_input) {
    if (is_input) {
      if (this->graph_viewer_.IsConstantInitializer(node_arg.Name(), true))
        return;
    }
    auto shape = node_arg.Shape();
    if (shape != nullptr) {
      // Can't have no dimensions
      if (shape->dim_size() == 0) {
        if (op_is_supported(optype, no_dimension_supported_)) {
          return;
        }
        if ((optype == "Identity") || (optype == "Sqrt")) {
          return;
        }
        has_unsupported_dimension = true;
        return;
      } else {
        // Zero dimension check
        for (const auto& dim : shape->dim()) {
          if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
            if ((device_id_.find("GPU") != std::string::npos) && ((optype == "Expand") ||
                                                                  (optype == "Slice") || (optype == "Concat") || (optype == "Shape"))) {
              return;
            }
            has_unsupported_dimension = true;
            return;
          }
        }
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
  if (domain == kOnnxDomain && unsupported_op_mode(node)) {
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
  const auto opset = op_map.find(domain);
  const auto op_fun = ops_supported_as_function.find(node->OpType());
  if (opset == op_map.end()) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Failed in Unsupported onnx model domain" << std::endl;
    }
#endif
    return false;
  }
  if (opset->second.find(optype) == opset->second.end() && op_fun == ops_supported_as_function.end()) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "The operator is not available in OpenVINO ngraph operators list nor the operator is a special ONNX function" << std::endl;
    }
#endif
    return false;
  }
  return true;
}

std::vector<NodeIndex> DataOps::GetUnsupportedNodeIndices(std::unordered_set<std::string>& ng_required_initializers) {
  const auto ng_supported_ops = GetNgSupportedOps(GetOnnxOpSet(graph_viewer_));

  std::vector<NodeIndex> unsupported_nodes_idx;

  for (const auto& node_idx : graph_viewer_.GetNodesInTopologicalOrder()) {
    if (node_is_supported(ng_supported_ops, node_idx)) {
      // Collect inputs that are initializers
      graph_viewer_.GetNode(node_idx)->ForEachDef([&ng_required_initializers, this](const NodeArg& node_arg, bool is_input) {
            if(is_input && this->graph_viewer_.GetAllInitializedTensors().count(node_arg.Name())) {
                ng_required_initializers.insert(node_arg.Name());
              } }, true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }
  return unsupported_nodes_idx;
}

bool DataOps::IsOpSupportedOnlyInModel(std::string name) {
  return ops_supported_only_in_model.find(name) != ops_supported_only_in_model.end();
}

bool DataOps::SpecialConditionForClusterSizeOne(std::unordered_set<std::string>& ng_required_initializers, const Node* node) {
  if (node->OpType() == "Reshape") {
    const auto& shape_arg = node->InputDefs()[1];
    if (ng_required_initializers.find(shape_arg->Name()) == ng_required_initializers.end()) {
      return true;
    }
  } else if (node->OpType() == "Expand") {
    // nGraph only supports constant shape input values
    const auto& output = node->OutputDefs()[0];
    if (output->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)
      return true;
  } else if (node->OpType() == "RoiAlign") {
    using onnx_dtype = ONNX_NAMESPACE::TensorProto_DataType;

    onnx_dtype input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype input_2_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[2]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

    if ((input_0_data_type != onnx_dtype::TensorProto_DataType_FLOAT16) ||
        (input_1_data_type != onnx_dtype::TensorProto_DataType_FLOAT16) ||
        (input_2_data_type != onnx_dtype::TensorProto_DataType_FLOAT) ||
        (output_data_type != onnx_dtype::TensorProto_DataType_FLOAT16))
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
