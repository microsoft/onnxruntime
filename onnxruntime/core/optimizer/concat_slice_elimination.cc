// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/concat_slice_elimination.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status ConcatSliceElimination::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_concat = graph.GetNode(node_index);
    if (p_concat == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& concat = *p_concat;
    ORT_RETURN_IF_ERROR(Recurse(concat, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(concat, "Concat", {4, 11, 13}, kOnnxDomain) ||
        !graph_utils::IsSupportedProvider(concat, GetCompatibleExecutionProviders())) {
      continue;
    }

    if (ConcatSliceElimination::FuseConcatSliceSubgraph(concat, graph, logger)) {
      fused_count++;
      modified = true;
    }
  }

  if (fused_count > 0) {
    LOGS(logger, INFO) << "Total fused concat node count: " << fused_count;
  }

  return Status::OK();
}

static bool GetSliceInfo(const Graph& graph,
                         const Node& node,
                         const logging::Logger& logger,
                         InlinedVector<int64_t>& starts,
                         InlinedVector<int64_t>& ends,
                         InlinedVector<int64_t>& axes,
                         InlinedVector<int64_t>& steps) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {1, 10, 11, 13})) {
    return false;
  }

  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  if (graph_utils::MatchesOpSinceVersion(node, {1})) {
    // If it is a Slice operator of opset version 1, starts/ends/axes are provided as node attributes.
    if (!graph_utils::GetRepeatedNodeAttributeValues(node, "starts", starts) ||
        !graph_utils::GetRepeatedNodeAttributeValues(node, "ends", ends) ||
        starts.size() != ends.size()) {
      return false;
    }
    // If there is an axes attribute, it has to be the same size as the starts and ends.
    if (graph_utils::GetRepeatedNodeAttributeValues(node, "axes", axes) && (axes.size() != starts.size())) {
      return false;
    }
  } else if (graph_utils::MatchesOpSinceVersion(node, {10, 11, 13})) {
    // If it is a Slice operator of opset version >= 10, starts/ends/axes/steps are provided as node inputs.

    // Returns a pointer to the corresponding NodeArg if input of the node at this index exists; otherwise, a nullptr.
    auto get_input_if_exists = [&node](size_t input_idx) -> const NodeArg* {
      const auto& input_defs = node.InputDefs();
      const NodeArg* input = (input_defs.size() > input_idx) ? input_defs[input_idx] : nullptr;
      return (input == nullptr || !input->Exists()) ? nullptr : input;
    };

    // Returns a pointer to the initializer if it is constant; otherwise, a nullptr.
    auto get_initializer_if_constant =
        [&graph, get_input_if_exists](size_t input_idx) -> const ONNX_NAMESPACE::TensorProto* {
      const NodeArg* input = get_input_if_exists(input_idx);
      return input ? graph_utils::GetConstantInitializer(graph, input->Name()) : nullptr;
    };

    auto get_initializer_data =
        [&graph](const ONNX_NAMESPACE::TensorProto* initializer) -> InlinedVector<int64_t> {
      Initializer init(*initializer, graph.ModelPath());
      if (initializer->data_type() == ONNX_NAMESPACE::TensorProto::INT32) {
        int32_t* init_data = init.data<int32_t>();
        return InlinedVector<int64_t>(init_data, init_data + init.size());
      } else if (initializer->data_type() == ONNX_NAMESPACE::TensorProto::INT64) {
        int64_t* init_data = init.data<int64_t>();
        return InlinedVector<int64_t>(init_data, init_data + init.size());
      }
      return {};
    };

    // Starts and ends inputs have to exist, be constant, and be of the same size.
    const ONNX_NAMESPACE::TensorProto* starts_init = get_initializer_if_constant(1);
    const ONNX_NAMESPACE::TensorProto* ends_init = get_initializer_if_constant(2);
    if (starts_init && ends_init) {
      starts = get_initializer_data(starts_init);
      ends = get_initializer_data(ends_init);

      if (starts.size() == 0 || ends.size() == 0 || starts.size() != ends.size()) {
        return false;
      }

      // If axes input exists, it should be constant and of the same size as starts/ends.
      if (get_input_if_exists(3)) {
        const ONNX_NAMESPACE::TensorProto* axes_init = get_initializer_if_constant(3);
        if (!axes_init || axes_init->dims_size() != 1 ||
            static_cast<size_t>(axes_init->dims().Get(0)) != starts.size()) {
          return false;
        }
        axes = get_initializer_data(axes_init);

        // If steps input exists, it should be constant and all values should be 1.
        if (get_input_if_exists(4)) {
          const ONNX_NAMESPACE::TensorProto* steps_init = get_initializer_if_constant(4);
          if (!steps_init) {
            return false;
          }
          steps = get_initializer_data(steps_init);
          if (steps.size() != starts.size()) {
            return false;
          }
          for (int64_t step : steps) {
            if (step != 1) {
              return false;
            }
          }
        }
      }
    } else {
      // Should be unreachable, but just to be safe in case a new op version is added.
      return false;
    }
  } else {
    return false;
  }
  return true;
}

/**
Apply Concat Slice Elimination transform. This transform removes the redundant
Concat + Slice pattern if the concat and slice axis is 0 and we are slicing the same
sizes as the inputs to concat.

Before transform:
ip_0(l0)               -- >Slice [0, l0] ---> op0
      \               |
ip_1(l1)-- Concat(axis=0) -->Slice [q, l0+l1]---> op1
      /               |
ip_2(l2)                -->Slice [l0+l1, :]---> op2

After transform:
ip_0 ---> op0

ip_1 ---> op1

ip_2 ---> op2

*/
bool ConcatSliceElimination::FuseConcatSliceSubgraph(Node& concat, Graph& graph, const logging::Logger& logger) {
  // The root could be either a graph input or a node so use node arg to compare.
  auto& concat_inputs = concat.MutableInputDefs();
  auto concat_outputs = graph.GetMutableConsumerNodes(concat.MutableOutputDefs()[0]->Name());

  // number of inputs and outputs must be equal
  if (concat_outputs.size() != concat_inputs.size()) return false;

  size_t num_inputs = concat_inputs.size();

  // inputs to concat must be initializers
  bool is_valid = true;
  for (size_t i = 0; i < num_inputs; i++) {
    is_valid = is_valid && graph_utils::IsInitializer(graph, concat_inputs[i]->Name(), true);
  }
  if (!is_valid) return false;

  // concat axis must be 0
  const auto* axis_attr = graph_utils::GetNodeAttribute(concat, "axis");
  if (axis_attr == nullptr || !utils::HasInt(*axis_attr) || axis_attr->i() != 0) {
    return false;
  }

  auto get_initializer_size =
      [&graph](const std::string& name) -> int64_t {
    const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
    if (!graph.GetInitializedTensor(name, initializer)) {
      return static_cast<int64_t>(-1);
    }
    int64_t size = 1;
    for (auto& dim : initializer->dims()) {
      size *= dim;
    }
    return size;
  };

  InlinedVector<int64_t> concat_input_len(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    concat_input_len[i] = get_initializer_size(concat_inputs[i]->Name());
    if (concat_input_len[i] == -1) {  // invalid size
      return false;
    }
  }

  InlinedVector<int64_t> cumulative_input_len(num_inputs + 1, 0);
  std::partial_sum(concat_input_len.begin(), concat_input_len.end(), cumulative_input_len.begin() + 1);
  InlinedVector<bool> visited(num_inputs, false);

  InlinedVector<onnxruntime::Node*> ordered_slice;
  ordered_slice.reserve(concat_outputs.size());
  ordered_slice.assign(concat_outputs.begin(), concat_outputs.end());

  for (auto slice : concat_outputs) {
    InlinedVector<int64_t> starts, ends, axes, steps;
    if (!GetSliceInfo(graph, *slice, logger, starts, ends, axes, steps)) return false;
    if (starts.size() > 1) return false;
    if (axes[0] != 0) return false;
    if (steps[0] != 1) return false;
    auto iter = std::find(cumulative_input_len.begin(), cumulative_input_len.end(), starts[0]);
    if (iter != cumulative_input_len.end()) {
      size_t idx = iter - cumulative_input_len.begin();
      if (visited[idx]) {
        // this start length has already been visited
        return false;
      }
      auto next = iter + 1;
      if (next != cumulative_input_len.end() && (*next == ends[0] || idx == num_inputs - 1)) {
        // found a match!
        ordered_slice[idx] = slice;
        visited[idx] = true;
      } else {  // no match for ends
        return false;
      }
    } else {  // no match for starts
      return false;
    }
  }
  int replace_cnt = 0;
  for (auto slice_node : concat_outputs) {
    Node& op_node = *graph.GetNode(slice_node->OutputNodesBegin()->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *slice_node);
    graph_utils::ReplaceNodeInput(op_node, 1, *(concat_inputs[replace_cnt]));
    replace_cnt++;
  }
  if (replace_cnt == 3) {
    LOGS(logger, INFO) << "Fused concat node: " << concat.OutputDefs()[0]->Name();

    // delete the slice nodes and concat node
    graph_utils::RemoveNodeOutputEdges(graph, concat);
    graph.RemoveNode(concat.Index());
    for (auto slice_node : ordered_slice) {
      graph.RemoveNode(slice_node->Index());
    }
    return true;
  }
  return false;
}

}  // namespace onnxruntime
