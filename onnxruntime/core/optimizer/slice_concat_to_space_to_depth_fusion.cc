// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/slice_concat_to_space_to_depth_fusion.h"

#include <array>
#include <limits>
#include <numeric>
#include <vector>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace {

using IntValues = InlinedVector<int64_t>;

struct SlicePhase {
  int64_t h_offset;
  int64_t w_offset;
};

struct NormalizedSliceParams {
  std::array<int64_t, 4> starts;
  std::array<int64_t, 4> ends;
  std::array<int64_t, 4> steps;
};

constexpr int64_t kRank = 4;
constexpr int64_t kChannelAxis = 1;
constexpr int64_t kHeightAxis = 2;
constexpr int64_t kWidthAxis = 3;
constexpr int64_t kBlockSize = 2;

int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

bool GetInitializerIntValues(const Graph& graph, const TensorProto* initializer, IntValues& values) {
  if (initializer == nullptr || initializer->dims_size() != 1) {
    return false;
  }

  Initializer init(graph, *initializer, graph.ModelPath());
  if (initializer->data_type() == TensorProto::INT32) {
    const int32_t* init_data = init.data<int32_t>();
    values.assign(init_data, init_data + init.size());
    return true;
  }

  if (initializer->data_type() == TensorProto::INT64) {
    const int64_t* init_data = init.data<int64_t>();
    values.assign(init_data, init_data + init.size());
    return true;
  }

  return false;
}

bool GetConstantInputIntValues(const Graph& graph, const NodeArg* input, IntValues& values) {
  if (input == nullptr || !input->Exists()) {
    return false;
  }

  if (const TensorProto* initializer = graph_utils::GetConstantInitializer(graph, input->Name()); initializer != nullptr) {
    return GetInitializerIntValues(graph, initializer, values);
  }

  const Node* producer = graph.GetProducerNode(input->Name());
  if (producer == nullptr || producer->OpType() != "Constant" || producer->Domain() != kOnnxDomain) {
    return false;
  }

  const auto& attributes = producer->GetAttributes();
  const auto attr_it = attributes.find("value");
  if (attr_it == attributes.end() || attr_it->second.type() != AttributeProto_AttributeType_TENSOR) {
    return false;
  }

  return GetInitializerIntValues(graph, &attr_it->second.t(), values);
}

bool GetSliceInfo(const Graph& graph,
                  const Node& node,
                  const logging::Logger& logger,
                  IntValues& starts,
                  IntValues& ends,
                  IntValues& axes,
                  IntValues& steps) {
  ORT_UNUSED_PARAMETER(logger);

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {10, 11, 13}, kOnnxDomain) ||
      graph.NodeProducesGraphOutput(node)) {
    return false;
  }

  auto get_input_if_exists = [&node](size_t input_idx) -> const NodeArg* {
    const auto& input_defs = node.InputDefs();
    const NodeArg* input = input_defs.size() > input_idx ? input_defs[input_idx] : nullptr;
    return (input == nullptr || !input->Exists()) ? nullptr : input;
  };

  if (!GetConstantInputIntValues(graph, get_input_if_exists(1), starts) ||
      !GetConstantInputIntValues(graph, get_input_if_exists(2), ends) ||
      starts.empty() || starts.size() != ends.size()) {
    return false;
  }

  axes.clear();
  steps.clear();

  if (const NodeArg* axes_input = get_input_if_exists(3); axes_input != nullptr) {
    if (!GetConstantInputIntValues(graph, axes_input, axes) || axes.size() != starts.size()) {
      return false;
    }
  } else {
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), int64_t{0});
  }

  if (const NodeArg* steps_input = get_input_if_exists(4); steps_input != nullptr) {
    if (!GetConstantInputIntValues(graph, steps_input, steps) || steps.size() != starts.size()) {
      return false;
    }
  } else {
    steps.assign(starts.size(), int64_t{1});
  }

  return true;
}

bool IsSupportedSpaceToDepthInputType(const NodeArg& input) {
  const auto* type_proto = input.TypeAsProto();
  if (type_proto == nullptr || !type_proto->has_tensor_type()) {
    return false;
  }

  const int32_t elem_type = type_proto->tensor_type().elem_type();

  // TODO(hasesh): Consider supporting float16 too ?
  if (elem_type != TensorProto::FLOAT && elem_type != TensorProto::DOUBLE) {
    return false;
  }

  const auto& tensor_type = type_proto->tensor_type();
  if (tensor_type.has_shape()) {
    return tensor_type.shape().dim_size() == kRank;
  }

  return true;
}

bool TryGetStaticChannelCount(const NodeArg& input, int64_t& channel_count) {
  const auto* type_proto = input.TypeAsProto();
  if (type_proto == nullptr || !type_proto->has_tensor_type()) {
    return false;
  }

  const auto& tensor_type = type_proto->tensor_type();
  if (!tensor_type.has_shape() || tensor_type.shape().dim_size() != kRank) {
    return false;
  }

  const auto& channel_dim = tensor_type.shape().dim(onnxruntime::narrow<int>(kChannelAxis));
  if (!utils::HasDimValue(channel_dim) || channel_dim.dim_value() <= 0) {
    return false;
  }

  channel_count = channel_dim.dim_value();
  return true;
}

bool TryGetStaticInputDim(const NodeArg& input, int64_t axis, int64_t& dim_value) {
  const auto* type_proto = input.TypeAsProto();
  if (type_proto == nullptr || !type_proto->has_tensor_type()) {
    return false;
  }

  const auto& tensor_type = type_proto->tensor_type();
  if (!tensor_type.has_shape() || tensor_type.shape().dim_size() != kRank || axis < 0 || axis >= kRank) {
    return false;
  }

  const auto& dim = tensor_type.shape().dim(onnxruntime::narrow<int>(axis));
  if (!utils::HasDimValue(dim) || dim.dim_value() <= 0) {
    return false;
  }

  dim_value = dim.dim_value();
  return true;
}

bool IsFullExtentEnd(const NodeArg& input, int64_t axis, int64_t end) {
  if (end == std::numeric_limits<int64_t>::max()) {
    return true;
  }

  if (end < 0) {
    return false;
  }

  int64_t dim_value = 0;
  return TryGetStaticInputDim(input, axis, dim_value) && end == dim_value;
}

TypeProto MakeSpaceToDepthOutputTypeProto(const NodeArg& input) {
  TypeProto output_type;

  const auto* input_type_proto = input.TypeAsProto();
  if (input_type_proto == nullptr) {
    return output_type;
  }

  output_type = *input_type_proto;

  if (!output_type.has_tensor_type()) {
    return output_type;
  }

  auto* output_shape = output_type.mutable_tensor_type()->mutable_shape();
  if (output_shape == nullptr || output_shape->dim_size() != kRank) {
    return output_type;
  }

  auto* channel_dim = output_shape->mutable_dim(onnxruntime::narrow<int>(kChannelAxis));
  if (utils::HasDimValue(*channel_dim) && channel_dim->dim_value() > 0) {
    channel_dim->set_dim_value(channel_dim->dim_value() * kBlockSize * kBlockSize);
  } else {
    channel_dim->clear_dim_value();
    channel_dim->clear_dim_param();
  }

  for (const int64_t axis : {kHeightAxis, kWidthAxis}) {
    auto* dim = output_shape->mutable_dim(onnxruntime::narrow<int>(axis));
    if (utils::HasDimValue(*dim) && dim->dim_value() > 0 && dim->dim_value() % kBlockSize == 0) {
      dim->set_dim_value(dim->dim_value() / kBlockSize);
    } else {
      dim->clear_dim_value();
      dim->clear_dim_param();
    }
  }

  return output_type;
}

bool TryMatchSlicePhase(const Graph& graph,
                        const Node& slice,
                        const NodeArg& common_input,
                        const logging::Logger& logger,
                        NormalizedSliceParams& params,
                        SlicePhase& phase) {
  if (slice.InputDefs().empty() || slice.InputDefs()[0] != &common_input) {
    return false;
  }

  IntValues starts;
  IntValues ends;
  IntValues axes;
  IntValues steps;
  if (!GetSliceInfo(graph, slice, logger, starts, ends, axes, steps)) {
    return false;
  }

  params.starts = {0, 0, 0, 0};
  params.ends = {
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max()};
  params.steps = {1, 1, 1, 1};
  std::array<bool, 4> axis_seen{false, false, false, false};

  for (size_t i = 0; i < starts.size(); ++i) {
    const int64_t axis = NormalizeAxis(axes[i], kRank);
    if (axis < 0 || axis >= kRank) {
      return false;
    }

    if (axis_seen[onnxruntime::narrow<size_t>(axis)]) {
      return false;
    }

    axis_seen[onnxruntime::narrow<size_t>(axis)] = true;

    params.starts[onnxruntime::narrow<size_t>(axis)] = starts[i];
    params.ends[onnxruntime::narrow<size_t>(axis)] = ends[i];
    params.steps[onnxruntime::narrow<size_t>(axis)] = steps[i];
  }

  for (size_t axis = 0; axis < params.starts.size(); ++axis) {
    if (params.starts[axis] < 0 || params.steps[axis] <= 0) {
      return false;
    }
  }

  if (params.starts[0] != 0 || params.starts[1] != 0 ||
      params.steps[0] != 1 || params.steps[1] != 1 ||
      params.steps[kHeightAxis] != kBlockSize || params.steps[kWidthAxis] != kBlockSize) {
    return false;
  }

  if (!IsFullExtentEnd(common_input, 0, params.ends[0]) ||
      !IsFullExtentEnd(common_input, 1, params.ends[1]) ||
      !IsFullExtentEnd(common_input, kHeightAxis, params.ends[kHeightAxis]) ||
      !IsFullExtentEnd(common_input, kWidthAxis, params.ends[kWidthAxis])) {
    return false;
  }

  const int64_t h_offset = params.starts[kHeightAxis];
  const int64_t w_offset = params.starts[kWidthAxis];
  if ((h_offset != 0 && h_offset != 1) || (w_offset != 0 && w_offset != 1)) {
    return false;
  }

  phase = {h_offset, w_offset};
  return true;
}

bool IsSingleConsumerOfConcat(const Graph& graph, const Node& slice, const Node& concat) {
  const auto consumers = graph.GetConsumerNodes(slice.OutputDefs()[0]->Name());
  return consumers.size() == 1 && consumers.front() == &concat;
}

bool TryGetPhasePermutation(const std::array<SlicePhase, 4>& actual_phases,
                            std::array<int64_t, 4>& permutation) {
  static constexpr std::array<SlicePhase, 4> kCanonicalPhases{{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};
  std::array<bool, 4> used{false, false, false, false};

  for (size_t i = 0; i < actual_phases.size(); ++i) {
    bool matched = false;
    for (size_t j = 0; j < kCanonicalPhases.size(); ++j) {
      if (!used[j] && actual_phases[i].h_offset == kCanonicalPhases[j].h_offset &&
          actual_phases[i].w_offset == kCanonicalPhases[j].w_offset) {
        permutation[i] = static_cast<int64_t>(j);
        used[j] = true;
        matched = true;
        break;
      }
    }

    if (!matched) {
      return false;
    }
  }

  return true;
}

NodeArg* CreateInt64Initializer(Graph& graph,
                                const std::vector<int64_t>& values,
                                const std::string& name) {
  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(name);
  initializer.add_dims(onnxruntime::narrow<int64_t>(values.size()));
  initializer.set_data_type(TensorProto::INT64);
  utils::SetRawDataInTensorProto(initializer,
                                 reinterpret_cast<const char*>(values.data()),
                                 values.size() * sizeof(int64_t));
  return &graph_utils::AddInitializerWithOrtValue(graph, initializer);
}

bool FuseSliceConcatToSpaceToDepth(Node& concat, Graph& graph, const logging::Logger& logger) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(concat, "Concat", {4, 11, 13}, kOnnxDomain) ||
      concat.InputDefs().size() != 4) {
    return false;
  }

  const auto* axis_attr = graph_utils::GetNodeAttribute(concat, "axis");
  if (axis_attr == nullptr || !utils::HasInt(*axis_attr)) {
    return false;
  }

  const int64_t concat_axis = NormalizeAxis(axis_attr->i(), kRank);
  if (concat_axis != kChannelAxis) {
    return false;
  }

  Node* slice_nodes[4]{};
  const NodeArg* common_input = nullptr;
  const auto& provider_type = concat.GetExecutionProviderType();
  NormalizedSliceParams reference_params{};
  std::array<SlicePhase, 4> actual_phases{};

  for (size_t i = 0; i < concat.InputDefs().size(); ++i) {
    const NodeArg* concat_input = concat.InputDefs()[i];
    if (concat_input == nullptr || !concat_input->Exists()) {
      return false;
    }

    Node* slice = graph.GetMutableProducerNode(concat_input->Name());
    if (slice == nullptr || slice == &concat || slice->GetExecutionProviderType() != provider_type ||
        !IsSingleConsumerOfConcat(graph, *slice, concat)) {
      return false;
    }

    if (i == 0) {
      common_input = slice->InputDefs()[0];
      if (common_input == nullptr || !IsSupportedSpaceToDepthInputType(*common_input)) {
        return false;
      }
    }

    ORT_ENFORCE(common_input != nullptr);

    NormalizedSliceParams current_params{};
    SlicePhase phase{};
    if (!TryMatchSlicePhase(graph, *slice, *common_input, logger, current_params, phase)) {
      return false;
    }

    actual_phases[i] = phase;

    if (i == 0) {
      reference_params = current_params;
    } else if (current_params.ends != reference_params.ends ||
               current_params.steps != reference_params.steps ||
               current_params.starts[0] != reference_params.starts[0] ||
               current_params.starts[1] != reference_params.starts[1]) {
      return false;
    }

    if (graph.NodeProducesGraphOutput(*slice)) {
      return false;
    }

    slice_nodes[i] = slice;
  }

  std::array<int64_t, 4> phase_permutation{};
  if (!TryGetPhasePermutation(actual_phases, phase_permutation)) {
    return false;
  }

  const bool is_canonical_order = phase_permutation == std::array<int64_t, 4>{0, 1, 2, 3};

  InlinedVector<NodeArg*> space_to_depth_outputs;
  if (is_canonical_order) {
    space_to_depth_outputs = {};
  } else {
    auto space_to_depth_output_type = MakeSpaceToDepthOutputTypeProto(*common_input);
    space_to_depth_outputs.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("space_to_depth_out"), &space_to_depth_output_type));
  }

  Node& space_to_depth = graph.AddNode(graph.GenerateNodeName("SpaceToDepth"),
                                       "SpaceToDepth",
                                       is_canonical_order ? "Fused Slice*4 + Concat into SpaceToDepth"
                                                          : "Fused Slice*4 + Concat into SpaceToDepth + channel permutation",
                                       {const_cast<NodeArg*>(common_input)},
                                       space_to_depth_outputs,
                                       nullptr,
                                       kOnnxDomain);
  space_to_depth.AddAttribute("blocksize", kBlockSize);
  space_to_depth.SetExecutionProviderType(provider_type);

  Node* replacement_end = &space_to_depth;
  if (!is_canonical_order) {
    int64_t channel_count = 0;
    if (!TryGetStaticChannelCount(*common_input, channel_count)) {
      return false;
    }

    InlinedVector<int64_t> gather_indices;
    gather_indices.reserve(onnxruntime::narrow<size_t>(channel_count * kBlockSize * kBlockSize));
    for (const int64_t source_block_index : phase_permutation) {
      for (int64_t c = 0; c < channel_count; ++c) {
        gather_indices.push_back(source_block_index * channel_count + c);
      }
    }

    NodeArg* gather_indices_arg = CreateInt64Initializer(
        graph,
        std::vector<int64_t>(gather_indices.begin(), gather_indices.end()),
        graph.GenerateNodeArgName("space_to_depth_gather_indices"));

    Node& gather = graph.AddNode(graph.GenerateNodeName("Gather"),
                                 "Gather",
                                 "Reorder SpaceToDepth channels to preserve Slice+Concat block order",
                                 {space_to_depth.MutableOutputDefs()[0], gather_indices_arg},
                                 {},
                                 nullptr,
                                 kOnnxDomain);
    gather.AddAttribute("axis", static_cast<int64_t>(kChannelAxis));
    gather.SetExecutionProviderType(provider_type);
    replacement_end = &gather;
  }

  // `FinalizeNodeFusion()` moves all input edges from the first fused node to
  // the replacement start node by matching input names. Slice nodes in this
  // pattern may take additional non-data inputs via `Constant` nodes
  // (starts/ends/axes/steps). Remove those auxiliary input edges from the
  // first slice so only the shared data input edge is transferred to
  // `SpaceToDepth`.
  {
    auto slice_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(*slice_nodes[0]);
    std::vector<graph_utils::GraphEdge> auxiliary_input_edges;
    auxiliary_input_edges.reserve(slice_input_edges.size());

    for (const auto& edge : slice_input_edges) {
      if (edge.arg_name != common_input->Name()) {
        auxiliary_input_edges.push_back(edge);
      }
    }

    graph_utils::GraphEdge::RemoveGraphEdges(graph, auxiliary_input_edges);
  }

  graph_utils::MoveAllNodeInputEdges(graph, *slice_nodes[0], space_to_depth);

  auto concat_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(concat);
  replacement_end->MutableOutputDefs() = concat.MutableOutputDefs();

  for (const auto& edge : concat_output_edges) {
    graph.AddEdge(replacement_end->Index(), edge.dst_node, 0, edge.dst_arg_index);
  }

  for (Node* node : {slice_nodes[0], slice_nodes[1], slice_nodes[2], slice_nodes[3], &concat}) {
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  LOGS(logger, INFO) << "Fused Slice+Concat downsample pattern into "
                     << (is_canonical_order ? "SpaceToDepth" : "SpaceToDepth + Gather")
                     << " node sequence starting at: " << space_to_depth.Name();
  return true;
}

}  // namespace

Status SliceConcatToSpaceToDepthFusion::ApplyImpl(Graph& graph,
                                                  bool& modified,
                                                  int graph_level,
                                                  const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr) {
      continue;
    }

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    if (FuseSliceConcatToSpaceToDepth(node, graph, logger)) {
      modified = true;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
