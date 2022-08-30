// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/slice_elimination.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

Status EliminateSlice::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool EliminateSlice::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  // We currently support elimination for Slice operator v1.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {1, 10, 11, 13})) {
    return false;
  }

  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  InlinedVector<int64_t> starts;
  InlinedVector<int64_t> ends;

  if (graph_utils::MatchesOpSinceVersion(node, {1})) {
    // If it is a Slice operator of opset version 1, starts/ends/axes are provided as node attributes.
    if (!graph_utils::GetRepeatedNodeAttributeValues(node, "starts", starts) ||
        !graph_utils::GetRepeatedNodeAttributeValues(node, "ends", ends) ||
        starts.size() != ends.size()) {
      return false;
    }
    InlinedVector<int64_t> axes;
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

        // If steps input exists, it should be constant and all values should be 1.
        if (get_input_if_exists(4)) {
          const ONNX_NAMESPACE::TensorProto* steps_init = get_initializer_if_constant(4);
          if (!steps_init) {
            return false;
          }
          InlinedVector<int64_t> steps = get_initializer_data(steps_init);
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
  }

  // For now eliminate slice operators if starts=0 and ends=MAX_INT.
  // TODO: Take into account the input's shape to get a tighter bound for the ends.
  for (size_t i = 0; i < starts.size(); ++i) {
    if (starts[i] != 0 || ends[i] < INT64_MAX) {
      return false;
    }
  }

  return true;
}

}  // namespace onnxruntime
