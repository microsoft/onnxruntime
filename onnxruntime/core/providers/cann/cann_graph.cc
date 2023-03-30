// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <map>
#include <set>

#include "core/providers/cann/cann_graph.h"

namespace onnxruntime {
namespace cann {

static int lower_bound = 8;   // Supported domain version lower bounds
static int upper_bound = 15;  // Supported domain version upper bounds

std::once_flag flag;

/**
 * This function will been changed with the evolution of ONNX and CANN
 * and will be replaced by the corresponding API provided by CANN in the future, probably.
 **/
std::vector<NodeIndex> SupportONNXModel(const GraphViewer& graph_viewer) {
  static std::set<std::string> cann_supported_ops = {
      "Abs", "Acos", "Acosh", "Add",
      "And", "ArgMax", "ArgMin", "Asin",
      "Asinh", "Atan", "Atanh", "AveragePool",
      "BatchNormalization", "BitShift", "Cast", "Ceil",
      "Celu", "Clip", "Compress", "Concat",
      "Constant", "ConstantOfShape", "Conv", "ConvTranspose",
      "Cos", "Cosh", "CumSum", "DepthToSpace",
      "Det", "Div", "Dropout", "Elu",
      "Equal", "Erf", "Exp", "Expand",
      "EyeLike", "Flatten", "Floor", "Gather",
      "GatherElements", "GatherND", "Gemm", "GlobalAveragePool",
      "GlobalLpPool", "GlobalMaxPool", "Greater", "GreaterOrEqual",
      "Hardmax", "HardSigmoid", "HardSwish", "Identity",
      "InstanceNormalization", "LeakyRelu", "Less",
      "LessOrEqual", "Log", "LogSoftmax", "LpNormalization",
      "LpPool", "LRN", "LSTM", "MatMul",
      "Max", "MaxPool", "MaxRoiPool", "MaxUnpool",
      "Mean", "MeanVarianceNormalization", "Min", "Mod",
      "Mul", "Multinomial", "Neg", "NonMaxSuppression",
      "NonZero", "Not", "OneHot", "Or",
      "Pad", "Pow", "PRelu", "RandomNormalLike",
      "RandomUniform", "RandomUniformLike", "Range", "Reciprocal",
      "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
      "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd",
      "ReduceSum", "ReduceSumSquare", "Relu", "Reshape",
      "Resize", "ReverseSequence", "RoiAlign", "Round",
      "Scatter", "ScatterElements", "ScatterND", "Selu",
      "Shape", "Shrink", "Sigmoid", "Sign",
      "Sin", "Sinh", "Size", "Slice",
      "Softmax", "SoftmaxCrossEntropyLoss", "Softplus", "Softsign",
      "SpaceToDepth", "Split", "Sqrt", "Squeeze",
      "Sub", "Sum", "Tanh", "TfIdfVectorizer",
      "ThresholdedRelu", "Tile", "TopK", "Transpose",
      "Unsqueeze", "Where", "Xor"};

  std::vector<NodeIndex> unsupported_nodes;

  int domain_version = graph_viewer.DomainToVersionMap().at(kOnnxDomain);
  for (const auto& index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto& node = graph_viewer.GetNode(index);

    if (node->Domain() != kOnnxDomain || domain_version < lower_bound || domain_version > upper_bound ||
        !cann_supported_ops.count(node->OpType())) {
      unsupported_nodes.push_back(index);
      continue;
    }

    // When the ONNX model is converted into an OM model, the Constant node will be automatically removed,
    // Therefore, the Constant nodes must be run on single operator inference mode.
    bool is_all_constant = true;
    for (const auto& input : node->InputDefs()) {
      if (!graph_viewer.GetAllInitializedTensors().count(input->Name())) {
        is_all_constant = false;
        break;
      }
    }
    if (is_all_constant) {
      unsupported_nodes.push_back(index);
    }
  }

  return unsupported_nodes;
}

Status ParserONNXModel(std::string string_model, ge::Graph& graph) {
  std::map<ge::AscendString, ge::AscendString> parser_params;
  CANN_GRAPH_RETURN_IF_ERROR(ge::aclgrphParseONNXFromMem(string_model.data(),
                                                         string_model.size(),
                                                         parser_params,
                                                         graph));

  return Status::OK();
}

Status BuildONNXModel(ge::Graph& graph, std::string input_shape, const char* soc_name, std::string file_name,
                      CANNExecutionProviderInfo& info, ge::ModelBufferData& model) {
  std::call_once(flag, [&soc_name, &info]() {
    std::map<ge::AscendString, ge::AscendString> options;
    options.emplace(ge::ir_option::SOC_VERSION, soc_name);

    if (!info.precision_mode.empty())
      options.emplace(ge::ir_option::PRECISION_MODE, info.precision_mode.c_str());
    if (!info.op_select_impl_mode.empty())
      options.emplace(ge::ir_option::OP_SELECT_IMPL_MODE, info.op_select_impl_mode.c_str());
    if (!info.optypelist_for_implmode.empty())
      options.emplace(ge::ir_option::OPTYPELIST_FOR_IMPLMODE, info.optypelist_for_implmode.c_str());

    CANN_CALL_THROW(ge::aclgrphBuildInitialize(options));
  });

  std::map<ge::AscendString, ge::AscendString> options;
  options.emplace(ge::ir_option::INPUT_SHAPE, input_shape.c_str());
  CANN_GRAPH_RETURN_IF_ERROR(ge::aclgrphBuildModel(graph, options, model));

  CANN_GRAPH_RETURN_IF_ERROR(ge::aclgrphSaveModel(file_name.c_str(), model));

  return Status::OK();
}

}  // namespace cann
}  // namespace onnxruntime
