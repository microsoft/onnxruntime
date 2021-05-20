// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_scale_fusion.h"

#include "onnx/defs/attr_proto_util.h"

#include "core/common/optional.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {
template <typename T>
struct ExtractScalarAsFloatDispatchTarget {
  Status operator()(const ONNX_NAMESPACE::TensorProto& tensor_proto, const Path& model_path, float& scalar_float) {
    T scalar;
    ORT_RETURN_IF_ERROR(utils::UnpackTensor(tensor_proto, model_path, &scalar, 1));
    scalar_float = static_cast<float>(scalar);
    return Status::OK();
  }
};

optional<float> GetScalarConstantInitializer(const Graph& graph, const NodeArg& node_arg) {
  const auto* initializer = graph_utils::GetConstantInitializer(graph, node_arg.Name());

  if (!initializer) {
    // not a constant
    return {};
  }

  const auto* shape = node_arg.Shape();
  ORT_ENFORCE(
      shape,
      "Constant initializer NodeArg shape should not be null. NodeArg: ", node_arg.Name());

  if (utils::GetTensorShapeFromTensorShapeProto(*shape).Size() != 1) {
    // not a scalar
    return {};
  }

  float scalar{};
  utils::MLTypeCallDispatcher<
      uint32_t, uint64_t, int32_t, int64_t, MLFloat16, float, double, BFloat16>
      dispatcher{initializer->data_type()};
  ORT_THROW_IF_ERROR(
      (dispatcher.InvokeRet<Status, ExtractScalarAsFloatDispatchTarget>(*initializer, graph.ModelPath(), scalar)));

  return {scalar};
}

// gets the scale value and its input index if node is a fusable scale (Mul or Div by scalar constant)
optional<std::pair<float, int>> GetScaleFromNode(
    const Graph& graph, const Node& scale_node,
    const std::unordered_set<std::string>& excluded_initializer_names) {
  const auto is_excluded_initializer =
      [&excluded_initializer_names](const NodeArg& node_arg) {
        return excluded_initializer_names.find(node_arg.Name()) != excluded_initializer_names.end();
      };

  if (graph_utils::IsSupportedOptypeVersionAndDomain(scale_node, "Div", {7, 13, 14})) {
    // (x / scale_reciprocal)
    const auto div_inputs = scale_node.InputDefs();
    ORT_ENFORCE(div_inputs.size() == 2);

    const int scale_reciprocal_arg_index = 1;
    const NodeArg& scale_reciprocal_node_arg = *div_inputs[scale_reciprocal_arg_index];

    if (is_excluded_initializer(scale_reciprocal_node_arg)) return {};

    const auto divisor = GetScalarConstantInitializer(graph, scale_reciprocal_node_arg);

    if (!divisor.has_value()) return {};

    return {std::make_pair(1.0f / divisor.value(), scale_reciprocal_arg_index)};
  }

  if (graph_utils::IsSupportedOptypeVersionAndDomain(scale_node, "Mul", {7, 13, 14})) {
    // (x * scale) or (scale * x)
    const auto mul_inputs = scale_node.InputDefs();
    ORT_ENFORCE(mul_inputs.size() == 2);

    for (int scale_arg_index = 0; scale_arg_index < 2; ++scale_arg_index) {
      const NodeArg& scale_node_arg = *mul_inputs[scale_arg_index];

      if (is_excluded_initializer(scale_node_arg)) continue;

      const auto multiplier = GetScalarConstantInitializer(graph, scale_node_arg);

      if (!multiplier.has_value()) continue;

      return {std::make_pair(multiplier.value(), scale_arg_index)};
    }

    return {};
  }

  return {};
}

struct ScaleMergeInfo {
  // the edge from the base node to the original node
  Node::EdgeConstIterator node_to_merge_edge;
  // the scale of the original node
  float scale;
  // the index of the input or output def on the original node
  // this def is moved to the fused node
  // for a leading scale (scale -> MatMul), it will be the unscaled input
  // for a trailing scale (MatMul -> scale), it will be the scaled output
  int node_to_merge_def_index;
  // the index of the input or output def on the fused node
  int fused_node_def_index;
};

std::vector<ScaleMergeInfo> GetInputNodeMerges(
    Graph& graph, Node& node,
    const std::unordered_set<std::string>& excluded_initializer_names) {
  std::vector<ScaleMergeInfo> input_node_merges{};
  for (auto input_edge = node.InputEdgesBegin(); input_edge != node.InputEdgesEnd(); ++input_edge) {
    const Node& input_node = input_edge->GetNode();

    if (input_node.GetExecutionProviderType() != node.GetExecutionProviderType()) continue;
    const auto scale_and_index = GetScaleFromNode(graph, input_node, excluded_initializer_names);
    if (!scale_and_index.has_value()) continue;

    // assume scale nodes have 2 input defs, so to_scale_index == 1 - scale_index
    ORT_ENFORCE(input_node.InputDefs().size() == 2 && scale_and_index.value().second < 2);
    const int to_scale_index = 1 - scale_and_index.value().second;

    input_node_merges.push_back(
        {input_edge,
         scale_and_index.value().first,
         to_scale_index,
         input_edge->GetDstArgIndex()});
  }
  return input_node_merges;
}

std::vector<ScaleMergeInfo> GetOutputNodeMerges(
    Graph& graph, Node& node,
    const std::unordered_set<std::string>& excluded_initializer_names) {
  if (!optimizer_utils::CheckOutputEdges(graph, node, 1)) {
    return {};
  }

  std::vector<ScaleMergeInfo> output_node_merges{};
  for (auto output_edge = node.OutputEdgesBegin(); output_edge != node.OutputEdgesEnd(); ++output_edge) {
    const Node& output_node = output_edge->GetNode();

    if (output_node.GetExecutionProviderType() != node.GetExecutionProviderType()) continue;
    const auto scale_and_index = GetScaleFromNode(graph, output_node, excluded_initializer_names);
    if (!scale_and_index.has_value()) continue;

    ORT_ENFORCE(output_node.OutputDefs().size() == 1);
    const int scaled_index = 0;

    output_node_merges.push_back(
        {output_edge,
         scale_and_index.value().first,
         scaled_index,
         output_edge->GetSrcArgIndex()});
  }
  return output_node_merges;
}

bool IsMatMulInputTypeSupported(const Node& node) {
  // if no matching key is present, any data type is allowed
  static const std::map<std::string, std::vector<std::string>> k_supported_data_types{
      {kCudaExecutionProvider, {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"}},
      {kRocmExecutionProvider, {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"}},
      {kCpuExecutionProvider, {"tensor(float)"}},
  };

  const auto it = k_supported_data_types.find(node.GetExecutionProviderType());
  return it == k_supported_data_types.end() || optimizer_utils::IsSupportedDataType(node, it->second);
}

Status ProcessNode(
    Graph& graph, Node& node, bool& modified,
    const std::unordered_set<std::string>& excluded_initializer_names,
    const std::unordered_set<std::string>& compatible_execution_providers) {
  if (!graph_utils::IsSupportedProvider(node, compatible_execution_providers)) {
    return Status::OK();
  }

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {9, 13}) &&
      !graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedMatMul", {1}, kMSDomain)) {
    return Status::OK();
  }

  if (!IsMatMulInputTypeSupported(node)) {
    return Status::OK();
  }

  const std::vector<ScaleMergeInfo> input_node_merges = GetInputNodeMerges(
      graph, node, excluded_initializer_names);
  const std::vector<ScaleMergeInfo> output_node_merges = GetOutputNodeMerges(
      graph, node, excluded_initializer_names);

  if (input_node_merges.empty() && output_node_merges.empty()) {
    return Status::OK();
  }

  NodeAttributes fused_node_attrs =
      node.OpType() == "FusedMatMul" ? node.GetAttributes() : NodeAttributes{};

  {
    ONNX_NAMESPACE::AttributeProto& alpha_attr = fused_node_attrs["alpha"];
    float total_scale = utils::HasFloat(alpha_attr) ? alpha_attr.f() : 1.0f;
    auto accumulate_scale = [&total_scale](const ScaleMergeInfo& fusion) {
      total_scale *= fusion.scale;
    };

    std::for_each(input_node_merges.begin(), input_node_merges.end(), accumulate_scale);
    std::for_each(output_node_merges.begin(), output_node_merges.end(), accumulate_scale);

    alpha_attr = ONNX_NAMESPACE::MakeAttribute("alpha", total_scale);
  }

  auto get_mutable_node_to_merge = [&graph](const ScaleMergeInfo& merge) -> Node& {
    return *graph.GetNode(merge.node_to_merge_edge->GetNode().Index());
  };

  std::vector<NodeArg*> fused_node_inputs = node.MutableInputDefs();
  for (const auto& input_node_merge : input_node_merges) {
    Node& input_node = get_mutable_node_to_merge(input_node_merge);
    fused_node_inputs[input_node_merge.fused_node_def_index] =
        input_node.MutableInputDefs()[input_node_merge.node_to_merge_def_index];
  }

  std::vector<NodeArg*> fused_node_outputs = node.MutableOutputDefs();
  for (const auto& output_node_merge : output_node_merges) {
    Node& output_node = get_mutable_node_to_merge(output_node_merge);
    fused_node_outputs[output_node_merge.fused_node_def_index] =
        output_node.MutableOutputDefs()[output_node_merge.node_to_merge_def_index];
  }

  Node& matmul_scale_node = graph.AddNode(
      graph.GenerateNodeName(node.Name() + "_FusedMatMulAndScale"),
      "FusedMatMul",
      "Fused MatMul and Scale",
      fused_node_inputs,
      fused_node_outputs,
      &fused_node_attrs,
      kMSDomain);

  matmul_scale_node.SetExecutionProviderType(node.GetExecutionProviderType());

  {
    std::vector<std::reference_wrapper<Node>> nodes_to_remove{node};

    for (const auto& input_node_merge : input_node_merges) {
      // remove merged input node's output edge
      auto input_node_edge = input_node_merge.node_to_merge_edge;
      Node& input_node = get_mutable_node_to_merge(input_node_merge);
      graph.RemoveEdge(
          input_node.Index(), node.Index(),
          input_node_edge->GetSrcArgIndex(), input_node_edge->GetDstArgIndex());

      // only remove merged input node if it has no more outputs
      if (!optimizer_utils::CheckOutputEdges(graph, input_node, 0)) continue;

      nodes_to_remove.push_back(input_node);
    }

    for (const auto& output_node_merge : output_node_merges) {
      nodes_to_remove.push_back(get_mutable_node_to_merge(output_node_merge));
    }

    for (Node& node_to_remove : nodes_to_remove) {
      graph_utils::RemoveNodeOutputEdges(graph, node_to_remove);
      graph.RemoveNode(node_to_remove.Index());
    }
  }

  modified = true;

  return Status::OK();
}

}  // namespace

Status MatMulScaleFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger)
    const {
  GraphViewer graph_viewer{graph};
  const auto node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_index : node_indices) {
    auto* node = graph.GetNode(node_index);
    if (!node) continue;

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    ORT_RETURN_IF_ERROR(ProcessNode(
        graph, *node, modified, excluded_initializer_names_, GetCompatibleExecutionProviders()));
  }

  return Status::OK();
}

}  // namespace onnxruntime
