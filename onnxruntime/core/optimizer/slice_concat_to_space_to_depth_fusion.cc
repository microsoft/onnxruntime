// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/slice_concat_to_space_to_depth_fusion.h"

#include <algorithm>
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
// This fusion currently only recognizes the common blocksize=2 pattern used by
// YOLO-style focus layers: 4 Slice nodes with offsets in {0,1}x{0,1}, step=2,
// followed by channel-axis Concat. The same idea generalizes to arbitrary
// blocksize b by matching b^2 Slice nodes with offsets in {0..b-1}x{0..b-1},
// step=b, and extending the phase-permutation/channel-reorder logic
// accordingly. For now we intentionally keep the implementation limited to the
// blocksize=2 case.
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

const Node* GetInputProducerNode(const Node& node, size_t input_index) {
  const int input_arg_index = onnxruntime::narrow<int>(input_index);
  for (auto edge_it = node.InputEdgesBegin(), edge_end = node.InputEdgesEnd(); edge_it != edge_end; ++edge_it) {
    if (edge_it->GetDstArgIndex() == input_arg_index) {
      return &edge_it->GetNode();
    }
  }

  return nullptr;
}

Node* GetMutableInputProducerNode(Graph& graph, Node& node, size_t input_index) {
  const Node* producer = GetInputProducerNode(node, input_index);
  return producer == nullptr ? nullptr : graph.GetNode(producer->Index());
}

bool HasSingleOutputEdgeToNode(const Node& node, const Node& consumer) {
  if (node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto edge_it = node.OutputEdgesBegin();
  return edge_it != node.OutputEdgesEnd() && &edge_it->GetNode() == &consumer;
}

bool GetConstantInputIntValues(const Graph& graph, const Node& node, size_t input_index, IntValues& values) {
  const auto& input_defs = node.InputDefs();
  const NodeArg* input = input_defs.size() > input_index ? input_defs[input_index] : nullptr;
  if (input == nullptr || !input->Exists()) {
    return false;
  }

  if (const TensorProto* initializer = graph_utils::GetConstantInitializer(graph, input->Name()); initializer != nullptr) {
    return GetInitializerIntValues(graph, initializer, values);
  }

  const Node* producer = GetInputProducerNode(node, input_index);
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

  if (!GetConstantInputIntValues(graph, node, 1, starts) ||
      !GetConstantInputIntValues(graph, node, 2, ends) ||
      starts.empty() || starts.size() != ends.size()) {
    return false;
  }

  axes.clear();
  steps.clear();

  if (const NodeArg* axes_input = get_input_if_exists(3); axes_input != nullptr) {
    if (!GetConstantInputIntValues(graph, node, 3, axes) || axes.size() != starts.size()) {
      return false;
    }
  } else {
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), int64_t{0});
  }

  if (const NodeArg* steps_input = get_input_if_exists(4); steps_input != nullptr) {
    if (!GetConstantInputIntValues(graph, node, 4, steps) || steps.size() != starts.size()) {
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

  const auto& tensor_type = type_proto->tensor_type();
  if (!tensor_type.has_shape() || tensor_type.shape().dim_size() != kRank) {
    return false;
  }

  const int32_t elem_type = tensor_type.elem_type();

  // TODO(hasesh): Consider supporting float16 too ?
  if (elem_type != TensorProto::FLOAT && elem_type != TensorProto::DOUBLE) {
    return false;
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
  return TryGetStaticInputDim(input, axis, dim_value) && end >= dim_value;
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

bool NormalizeSliceParams(const IntValues& starts,
                          const IntValues& ends,
                          const IntValues& axes,
                          const IntValues& steps,
                          NormalizedSliceParams& params) {
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

    const size_t axis_index = onnxruntime::narrow<size_t>(axis);
    if (axis_seen[axis_index]) {
      return false;
    }

    axis_seen[axis_index] = true;

    params.starts[axis_index] = starts[i];
    params.ends[axis_index] = ends[i];
    params.steps[axis_index] = steps[i];
  }

  for (size_t axis = 0; axis < params.starts.size(); ++axis) {
    if (params.starts[axis] < 0 || params.steps[axis] <= 0) {
      return false;
    }
  }

  return true;
}

// A focus branch is not always a single Slice that strides both spatial axes.
// Exporters commonly emit one Slice per axis, so a branch can be a chain of two
// Slice nodes. Two chained nodes is enough to cover height and width.
constexpr size_t kMaxSliceChainLength = 2;

struct SliceChain {
  // Ordered from the Concat backwards, so index 0 feeds the Concat.
  InlinedVector<Node*> nodes;
  const NodeArg* root_input;
  SlicePhase phase;
};

// Walks back from one Concat input through up to kMaxSliceChainLength Slice
// nodes, accumulating which spatial axis each one strides. Succeeds only when
// height and width are each strided exactly once and every other axis passes
// through untouched.
bool TryMatchSliceChain(Graph& graph,
                        Node& concat,
                        size_t input_index,
                        const std::string& provider_type,
                        const logging::Logger& logger,
                        SliceChain& chain) {
  chain.nodes.clear();
  chain.root_input = nullptr;

  std::array<bool, 4> axis_strided{};
  std::array<int64_t, 4> axis_offset{};

  Node* node = GetMutableInputProducerNode(graph, concat, input_index);
  const Node* consumer = &concat;

  while (chain.nodes.size() < kMaxSliceChainLength) {
    if (node == nullptr || node == &concat ||
        node->GetExecutionProviderType() != provider_type ||
        graph.NodeProducesGraphOutput(*node)) {
      return false;
    }

    // The node feeding the Concat must be ours alone. A node further upstream
    // may legitimately be shared with a sibling branch, which is how a focus
    // layer that splits height once and width twice is usually exported.
    if (chain.nodes.empty()) {
      if (!HasSingleOutputEdgeToNode(*node, *consumer)) {
        return false;
      }
    } else if (node->GetOutputEdgesCount() == 0) {
      return false;
    }

    IntValues starts;
    IntValues ends;
    IntValues axes;
    IntValues steps;
    if (!GetSliceInfo(graph, *node, logger, starts, ends, axes, steps)) {
      return false;
    }

    NormalizedSliceParams params{};
    if (!NormalizeSliceParams(starts, ends, axes, steps, params)) {
      return false;
    }

    if (node->InputDefs().empty()) {
      return false;
    }

    const NodeArg* node_input = node->InputDefs()[0];
    if (node_input == nullptr || !node_input->Exists()) {
      return false;
    }

    // Batch and channel must pass through untouched.
    for (const int64_t axis : {int64_t{0}, kChannelAxis}) {
      const size_t axis_index = onnxruntime::narrow<size_t>(axis);
      if (params.starts[axis_index] != 0 || params.steps[axis_index] != 1 ||
          !IsFullExtentEnd(*node_input, axis, params.ends[axis_index])) {
        return false;
      }
    }

    bool strided_here = false;

    for (const int64_t axis : {kHeightAxis, kWidthAxis}) {
      const size_t axis_index = onnxruntime::narrow<size_t>(axis);

      if (!IsFullExtentEnd(*node_input, axis, params.ends[axis_index])) {
        return false;
      }

      if (params.steps[axis_index] == kBlockSize) {
        if (axis_strided[axis_index] ||
            (params.starts[axis_index] != 0 && params.starts[axis_index] != 1)) {
          return false;
        }

        axis_strided[axis_index] = true;
        axis_offset[axis_index] = params.starts[axis_index];
        strided_here = true;
      } else if (params.steps[axis_index] != 1 || params.starts[axis_index] != 0) {
        return false;
      }
    }

    if (!strided_here) {
      return false;
    }

    chain.nodes.push_back(node);

    const size_t height_index = onnxruntime::narrow<size_t>(kHeightAxis);
    const size_t width_index = onnxruntime::narrow<size_t>(kWidthAxis);

    if (axis_strided[height_index] && axis_strided[width_index]) {
      chain.root_input = node_input;
      chain.phase = {axis_offset[height_index], axis_offset[width_index]};
      return true;
    }

    consumer = node;
    node = GetMutableInputProducerNode(graph, *node, 0);
  }

  return false;
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

  const NodeArg* common_input = nullptr;
  const auto& provider_type = concat.GetExecutionProviderType();
  std::array<SliceChain, 4> chains{};
  std::array<SlicePhase, 4> actual_phases{};

  for (size_t i = 0; i < concat.InputDefs().size(); ++i) {
    const NodeArg* concat_input = concat.InputDefs()[i];
    if (concat_input == nullptr || !concat_input->Exists()) {
      return false;
    }

    if (!TryMatchSliceChain(graph, concat, i, provider_type, logger, chains[i])) {
      return false;
    }

    if (i == 0) {
      common_input = chains[i].root_input;
      if (common_input == nullptr || !IsSupportedSpaceToDepthInputType(*common_input)) {
        return false;
      }
    } else if (chains[i].root_input != common_input) {
      // Every branch has to strip the same tensor.
      return false;
    }

    actual_phases[i] = chains[i].phase;
  }

  ORT_ENFORCE(common_input != nullptr);

  std::array<int64_t, 4> phase_permutation{};
  if (!TryGetPhasePermutation(actual_phases, phase_permutation)) {
    return false;
  }

  const bool is_canonical_order = phase_permutation == std::array<int64_t, 4>{0, 1, 2, 3};
  int64_t channel_count = 0;
  if (!is_canonical_order && !TryGetStaticChannelCount(*common_input, channel_count)) {
    return false;
  }

  InlinedVector<NodeArg*> space_to_depth_outputs;
  if (is_canonical_order) {
    space_to_depth_outputs = {};
  } else {
    auto space_to_depth_output_type = MakeSpaceToDepthOutputTypeProto(*common_input);
    space_to_depth_outputs.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("space_to_depth_out"), &space_to_depth_output_type));
  }

  NodeArg* space_to_depth_input = graph.GetNodeArg(common_input->Name());

  Node& space_to_depth = graph.AddNode(graph.GenerateNodeName("SpaceToDepth"),
                                       "SpaceToDepth",
                                       is_canonical_order ? "Fused Slice*4 + Concat into SpaceToDepth"
                                                          : "Fused Slice*4 + Concat into SpaceToDepth + channel permutation",
                                       {space_to_depth_input},
                                       space_to_depth_outputs,
                                       concat,
                                       nullptr,
                                       kOnnxDomain);
  space_to_depth.AddAttribute("blocksize", kBlockSize);
  space_to_depth.SetExecutionProviderType(provider_type);

  Node* replacement_end = &space_to_depth;
  if (!is_canonical_order) {
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
                                 concat,
                                 nullptr,
                                 kOnnxDomain);
    gather.AddAttribute("axis", static_cast<int64_t>(kChannelAxis));
    gather.SetExecutionProviderType(provider_type);
    graph.AddEdge(space_to_depth.Index(), gather.Index(), 0, 0);
    replacement_end = &gather;
  }

  // Explicitly transfer the shared data-input edge from the topmost Slice of
  // the first branch to SpaceToDepth. This avoids
  // graph_utils::MoveAllNodeInputEdges(), which is not defined in extended
  // minimal builds.
  {
    Node& chain_root = *chains[0].nodes.back();
    const auto data_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(chain_root, 0);
    if (!data_input_edges.empty()) {
      ORT_ENFORCE(data_input_edges.size() == 1, "Expected a single data input edge for Slice node.");
      const auto& data_input_edge = data_input_edges[0];
      graph.AddEdge(data_input_edge.src_node, space_to_depth.Index(), data_input_edge.src_arg_index, 0);
    }
  }

  auto concat_output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(concat);
  replacement_end->MutableOutputDefs() = concat.MutableOutputDefs();

  for (const auto& edge : concat_output_edges) {
    graph.AddEdge(replacement_end->Index(), edge.dst_node, 0, edge.dst_arg_index);
  }

  // Drop the Concat and the Slice nodes that fed it directly, then any upstream
  // Slice left without consumers. Upstream nodes can be shared between two
  // branches, so they are deduplicated and only removed once orphaned.
  InlinedVector<Node*> upstream_nodes;

  graph_utils::RemoveNodeOutputEdges(graph, concat);
  graph.RemoveNode(concat.Index());

  for (const SliceChain& chain : chains) {
    for (size_t i = 1; i < chain.nodes.size(); ++i) {
      if (std::find(upstream_nodes.begin(), upstream_nodes.end(), chain.nodes[i]) == upstream_nodes.end()) {
        upstream_nodes.push_back(chain.nodes[i]);
      }
    }
  }

  for (const SliceChain& chain : chains) {
    Node& node = *chain.nodes[0];
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());
  }

  for (Node* node : upstream_nodes) {
    if (node->GetOutputEdgesCount() == 0 && !graph.NodeProducesGraphOutput(*node)) {
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    }
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
  bool local_modified = false;

  do {
    local_modified = false;

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
        local_modified = true;
        break;
      }
    }
  } while (local_modified);

  return Status::OK();
}

}  // namespace onnxruntime
