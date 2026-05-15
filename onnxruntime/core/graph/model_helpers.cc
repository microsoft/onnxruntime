// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/graph/model_helpers.h"

#include <string>
#include <string_view>
#include <vector>

#include "core/graph/function_utils.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

namespace {

void CollectLocalFunctionCalls(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
    InlinedHashSet<std::string_view>& seen_calls,
    InlinedVector<std::string_view>& called_functions);

void CollectLocalFunctionCalls(
    const ONNX_NAMESPACE::GraphProto& graph,
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
    InlinedHashSet<std::string_view>& seen_calls,
    InlinedVector<std::string_view>& called_functions) {
  for (const auto& node : graph.node()) {
    CollectLocalFunctionCalls(node, model_local_functions, seen_calls, called_functions);
  }
}

void CollectLocalFunctionCalls(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
    InlinedHashSet<std::string_view>& seen_calls,
    InlinedVector<std::string_view>& called_functions) {
  const auto function_id = function_utils::GetFunctionIdentifier(node.domain(), node.op_type(), node.overload());
  auto it = model_local_functions.find(function_id);
  if (it != model_local_functions.end()) {
    // Use string_view into the map key (stable storage).
    std::string_view key_view = it->first;
    if (seen_calls.insert(key_view).second) {
      called_functions.push_back(key_view);
    }
  }

  for (const auto& attr : node.attribute()) {
    if (attr.has_g()) {
      CollectLocalFunctionCalls(attr.g(), model_local_functions, seen_calls, called_functions);
    }

    for (const auto& graph : attr.graphs()) {
      CollectLocalFunctionCalls(graph, model_local_functions, seen_calls, called_functions);
    }
  }
}

}  // namespace

Status BuildLocalFunctionCallGraph(
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions,
    LocalFunctionCallGraph& call_graph) {
  call_graph.reserve(model_local_functions.size());

  for (const auto& [function_id, function_proto] : model_local_functions) {
    if (function_proto == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Null function proto for function id: ", function_id);
    }

    InlinedHashSet<std::string_view> seen_calls;
    InlinedVector<std::string_view> callees;
    for (const auto& node : function_proto->node()) {
      CollectLocalFunctionCalls(node, model_local_functions, seen_calls, callees);
    }

    call_graph.emplace(std::string_view(function_id), std::move(callees));
  }

  return Status::OK();
}

Status ValidateCallGraphAcyclic(const LocalFunctionCallGraph& call_graph) {
  enum class VisitState { kNotVisited,
                          kVisiting,
                          kVisited };

  InlinedHashMap<std::string_view, VisitState> visit_states;
  visit_states.reserve(call_graph.size());
  for (const auto& [function_id, _] : call_graph) {
    ORT_UNUSED_PARAMETER(_);
    visit_states.emplace(function_id, VisitState::kNotVisited);
  }

  // Each frame records the function being visited and a pointer to its callees vector
  // in the call graph (no per-frame allocation).
  struct DfsFrame {
    std::string_view function_id;
    const InlinedVector<std::string_view>* callees;
    size_t next_callee_index;
  };

  std::vector<DfsFrame> dfs_stack;

  for (const auto& [root_id, root_callees] : call_graph) {
    if (visit_states[root_id] == VisitState::kVisited) {
      continue;
    }

    visit_states[root_id] = VisitState::kVisiting;
    dfs_stack.push_back({root_id, &root_callees, 0});

    while (!dfs_stack.empty()) {
      auto& frame = dfs_stack.back();

      if (frame.next_callee_index >= frame.callees->size()) {
        // All callees processed — mark as fully visited and pop.
        visit_states[frame.function_id] = VisitState::kVisited;
        dfs_stack.pop_back();
        continue;
      }

      std::string_view callee_id = (*frame.callees)[frame.next_callee_index];
      frame.next_callee_index++;

      auto callee_state_it = visit_states.find(callee_id);
      if (callee_state_it == visit_states.end()) {
        // Callee not in the graph — skip.
        continue;
      }

      if (callee_state_it->second == VisitState::kVisited) {
        continue;
      }

      if (callee_state_it->second == VisitState::kVisiting) {
        // Cycle detected. Build cycle description from the stack.
        std::string cycle;
        bool in_cycle = false;
        for (const auto& f : dfs_stack) {
          if (f.function_id == callee_id) {
            in_cycle = true;
          }
          if (in_cycle) {
            if (!cycle.empty()) {
              cycle.append(" -> ");
            }
            cycle.append(f.function_id);
          }
        }
        cycle.append(" -> ");
        cycle.append(callee_id);

        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                               "Model local function definitions must not be recursive. Cycle detected: ", cycle);
      }

      // Push callee onto the DFS stack.
      auto callee_graph_it = call_graph.find(callee_id);
      if (callee_graph_it == call_graph.end()) {
        continue;
      }

      callee_state_it->second = VisitState::kVisiting;
      dfs_stack.push_back({callee_id, &callee_graph_it->second, 0});
    }
  }

  return Status::OK();
}

Status ValidateModelLocalFunctionAcyclic(
    const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& model_local_functions) {
  LocalFunctionCallGraph call_graph;
  ORT_RETURN_IF_ERROR(BuildLocalFunctionCallGraph(model_local_functions, call_graph));
  return ValidateCallGraphAcyclic(call_graph);
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
