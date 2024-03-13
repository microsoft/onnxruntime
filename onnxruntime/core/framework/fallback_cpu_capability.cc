// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/tensorprotoutils.h"
#include "core/common/inlined_containers.h"

#include <cstring>
#include <cstdlib>
#include <queue>

#include "onnx/defs/data_type_utils.h"

#include "core/framework/op_kernel.h"
#include "core/framework/utils.h"

using namespace ONNX_NAMESPACE::Utils;

namespace onnxruntime {

namespace {
constexpr int64_t kSmallInitializerThreshold = 100;

static bool IsSmallInitializer(const onnxruntime::GraphViewer& graph, const NodeArg* arg) {
  // 'true' in the function call is to let the searching for the initializer
  // continue in the outer scopes of the current (sub-)graph if applicable
  const ONNX_NAMESPACE::TensorProto* initializer_tensor =
      graph.GetGraph().GetInitializer(arg->Name(), true);

  // Not an initializer at all
  if (initializer_tensor == nullptr) {
    return false;
  }

  // Check if "small" enough
  int64_t size = 1;
  for (auto& dim : initializer_tensor->dims()) {
    size *= dim;
  }

  return size <= kSmallInitializerThreshold;
}
}  // namespace

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)
static InlinedHashSet<NodeIndex> GetShapeRelatedNodes(const onnxruntime::GraphViewer& viewer) {
  // Conceptually, this function traverse from shape-consuming nodes
  // to fallback all its upstream nodes to CPU. Consider a graph
  //
  //
  // The traversal should stop when
  //  1. hitting Shape, Size nodes, graph inputs, or graph initializers.
  //  2. [TODO] hitting nodes with some large inputs or outputs. Before that,
  //     we need shape inference to determine the size of the inputs and outputs.
  //     Some graph transforms add nodes without shape information, so
  //     checking shapes will make the algorithm more unstable now.
  LOGS_DEFAULT(VERBOSE) << "Call GetShapeRelatedNodes to identify extra CPU nodes." << std::endl;

  const static InlinedHashMap<std::string_view, InlinedHashMap<int64_t, std::vector<size_t>>> shape_related_inputs_in_nodes = {
      // 2nd input of Expand-13 is a shape-related input.
      {"Expand", {{13 /* since version */, {1} /* shape inputs' indices */}}},
      // 2nd input (indexed by 1) of Reshape-13, Reshape-14, Reshape-19, Reshape-21 is a shape-related input.
      {"Reshape", {{13, {1}}, {14, {1}}, {19, {1}}, {21, {1}}}},
      // 2nd input of Unsqueeze-13 and Unsqueeze-21 is a shape-related input.
      {"Unsqueeze", {{13, {1}}, {21, {1}}}},
      // 1st input of ConstantOfShape is a shape-related input.
      {"ConstantOfShape", {{9, {0}}, {20, {0}}, {21, {0}}}},
      // 2nd to 5th inputs of Slice-13 are shape-related inputs.
      {"Slice", {{13, {1, 2, 3, 4}}}}};

  auto& graph = viewer.GetGraph();
  // Each shape-producing node produces a tensor consumed
  // as shape, axis, size, and indices.
  // E.g.,
  //  shape = onnx::Concat(s0, s1)
  //  reshaped = onnx::Reshape(x, shape)
  // Then, the shape-producing node is Concat.
  InlinedHashSet<const Node*> shape_producing_nodes;
  // This loop collects all shape-producing nodes by finding
  // all nodes that produce tensors specified in shape_related_inputs_in_nodes.
  // E.g., for the above example, Concat is a shape-producing node because
  // "Reshape" has a shape-related input at index 1.
  for (auto& node : graph.Nodes()) {
    LOGS_DEFAULT(VERBOSE) << "Check if node " << node.Name() << " can be sink of shape sub-graph." << std::endl;
    auto op_type_it = shape_related_inputs_in_nodes.find(node.OpType());
    if (op_type_it == shape_related_inputs_in_nodes.end()) {
      // This node doesn't consume tensor as shape,
      // so we won't find any shape-producing node from it.
      continue;
    }
    auto op_type_version_it = op_type_it->second.find(node.SinceVersion());
    if (op_type_version_it == op_type_it->second.end()) {
      // This node doesn't consume tensor as shape in this version,
      // so we won't find any shape-producing node from it.
      continue;
    }

    // shape-like inputs' indices in this node.
    // E.g., for Reshape, it's [1] and for Slice, it's [1, 2, 3, 4].
    auto& shape_input_indices = op_type_version_it->second;
    // Now, this `node` is a shape-consuming node as defined by shape_related_inputs_in_nodes.
    // Let's find producers for shape-like tensors consumed by this `node`.
    // Consider this graph:
    //  shape = onnx::Concat(s0, s1)
    //  reshaped = onnx::Reshape(x, shape)
    // The loop below does:
    //  1. checks all `Reshape`'s inputs, `x` and `shape`,
    //  2. finds `shape` is a shape-related variable since Reshape's 2nd input is a shape-related input,
    //  3. and then records the producer of `shape` (i.e., `Concat`).
    for (auto& input_index : shape_input_indices) {
      auto input = node.InputDefs().at(input_index);
      auto producer_node = graph.GetProducerNode(input->Name());
      if (producer_node != nullptr && producer_node->OpType() != "Shape" && producer_node->OpType() != "Size") {
        // Assume shape-computing sub-graphs begins with Shape, Size, or graph inputs.
        // We should not fallback those nodes's upstream nodes to CPU; otherwise,
        // it may change
        //   GPU-tensor-x -> Mul -> GPU-tensor-y -> Shape -> CPU-tensor
        // to
        //   CPU-tensor-x -> Mul -> CPU-tensor -> Shape -> CPU-tensor
        // and slows down the computation.

        // After this for-loop, we will reversely traverse all nodes from every shape-producing node
        // found here until hitting Shape, Size nodes, graph inputs, or graph initializers.
        // All nodes on the traversal path will be forced to run on CPU.
        LOGS_DEFAULT(VERBOSE) << "Find a shape producing node (i.e., a node produces a tensor consumed as shape-like input in other nodes): " << node.Name() << std::endl;
        shape_producing_nodes.insert(producer_node);
      }
    }
  }

  InlinedHashSet<NodeIndex> shape_related_node_indices;
  for (auto& node : shape_producing_nodes) {
    LOGS_DEFAULT(VERBOSE) << "Begin the (topologically reverse) traversing from shape producing node: " << node->Name() << std::endl;
    std::vector<const Node*> start_nodes = {node};

    auto to_stop = [](const Node* n1, const Node* n2) {
      LOGS_DEFAULT(VERBOSE) << "Skip the traversal from " << n1->Name() << " to " << n2->Name() << " since " << n2->Name() << " is a Shape or Size node." << std::endl;
      return n2->OpType() == "Shape" || n2->OpType() == "Size";
    };

    // Reversely traverse all nodes from the shape-producing node.
    // Force nodes to be run on CPU when all inputs and outputs are small.
    // Stop the traversal when a "Shape" node is found.
    graph.ReverseDFSFrom(
        start_nodes,
        [&shape_related_node_indices](const Node* n) {
          LOGS_DEFAULT(VERBOSE) << "Find an upstream node in shape sub-graph (let's fallback it to CPU): " << n->Name() << std::endl;
          shape_related_node_indices.insert(n->Index());
        },
        nullptr,
        NodeCompare(),
        to_stop);
  }

  return shape_related_node_indices;
}
#endif

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)
std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                   const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                   gsl::span<const NodeIndex> tentative_nodes,
                                                   const bool aggressive_cpu_fallback) {
#else
std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                   const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                   gsl::span<const NodeIndex> tentative_nodes) {
#endif
  // automatic conversion from const std::vector&
  const auto& ordered_nodes = graph.GetNodesInTopologicalOrder();
  InlinedVector<size_t> node_id_to_order_map(graph.MaxNodeIndex());
  for (size_t id = 0, limit = ordered_nodes.size(); id < limit; ++id) {
    const NodeIndex& node_id = ordered_nodes[id];
    node_id_to_order_map[node_id] = id;
  }

  // If return false, n1 will be output first; If return true, n2 will be output first
  auto greater_order_comp = [&](const NodeIndex n1, const NodeIndex n2) {
    return node_id_to_order_map[n1] > node_id_to_order_map[n2];
  };

  std::priority_queue<NodeIndex, std::vector<NodeIndex>, decltype(greater_order_comp)> candidates(greater_order_comp);

  InlinedHashSet<const NodeArg*> cpu_output_args;

  InlinedHashSet<NodeIndex> provider_nodes;
  provider_nodes.reserve(tentative_nodes.size());

  InlinedHashMap<NodeIndex, const KernelCreateInfo*> node_to_kernel;
  node_to_kernel.reserve(tentative_nodes.size());

  for (auto& node_id : tentative_nodes) {
    provider_nodes.insert(node_id);
    const Node* node = graph.GetNode(node_id);

    const KernelCreateInfo* kernel_info = kernel_lookup.LookUpKernel(*node);
    // at least one registry has a target provider's kernel for this node
    ORT_ENFORCE(kernel_info != nullptr);
    node_to_kernel.insert({node_id, kernel_info});

    // first, find all the direct consumer of cpu tensors.
    ORT_THROW_IF_ERROR(node->ForEachWithIndex(
        node->OutputDefs(),
        [&](const NodeArg& node_arg, size_t out_index) {
          if (utils::IsOutputOnCpu(*node, kernel_info, out_index)) {
            cpu_output_args.insert(&node_arg);
            auto consumer_nodes = graph.GetConsumerNodes(node_arg.Name());
            for (auto& consumer_node : consumer_nodes) {
              candidates.push(consumer_node->Index());
              LOGS_DEFAULT(VERBOSE) << "Candidate for fallback CPU execution: " << consumer_node->Name();
            }
          }
          return Status::OK();
        }));
  }

  const auto& graph_inputs = graph.GetInputs();
  InlinedHashSet<NodeIndex> visited;
  visited.reserve(candidates.size());
  std::unordered_set<NodeIndex> cpu_nodes;
  cpu_nodes.reserve(candidates.size());
  // The algo below is trying to identity a subgraph that only depends on cpu tensors.
  // Usually it is a subgraph that doing shape calculation based on a GPU tensor, then reshape it back.
  // The detail:
  // for each candidate, if one of its input is a cpu tensor and the Non-CPU kernel doesn't mark it as cpu input,
  // force the node to CPU to avoid memory cpu and add its output to the small cpu tensors.
  while (!candidates.empty()) {
    NodeIndex cur = candidates.top();
    candidates.pop();

    auto p = visited.insert(cur);
    if (!p.second)
      continue;

    auto* node = graph.GetNode(cur);
    if (provider_nodes.find(cur) == provider_nodes.end()) {
      // Nodes not in provider_nodes are either have EP assigned or no kernel found on target EP.
      // we assume these nodes will fallback to CPU, so add all direct consumers of all outputs to candidates.
      if (node->GetExecutionProviderType().empty() || node->GetExecutionProviderType() == kCpuExecutionProvider) {
        for (auto* output : node->OutputDefs()) {
          cpu_output_args.insert(output);
        }
        for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
          candidates.push((*it).Index());
        }
      }
      continue;
    }

    bool place_in_cpu = true;
    for (size_t i = 0; i < node->InputDefs().size(); ++i) {
      auto* input = node->InputDefs()[i];

      // skip placing on CPU if the data typs is float16 or bfloat16 or float8e4m3fn, float8e4m3fnuz, floate5m2, floate5m2fnuz
      if (input->Type() == DataTypeUtils::ToType("float16") ||
          input->Type() == DataTypeUtils::ToType("bfloat16") ||
          input->Type() == DataTypeUtils::ToType("float8e4m3fn") ||
          input->Type() == DataTypeUtils::ToType("float8e4m3fnuz") ||
          input->Type() == DataTypeUtils::ToType("float8e5m2") ||
          input->Type() == DataTypeUtils::ToType("float8e5m2fnuz")) {
        place_in_cpu = false;
        break;
      }

      // allow placing on CPU if it's a small initializer or graph input
      if (IsSmallInitializer(graph, input) ||
          std::find(graph_inputs.begin(), graph_inputs.end(), input) != graph_inputs.end()) {
        continue;
      }

      // the input is not a CPU tensor
      if (cpu_output_args.find(input) == cpu_output_args.end()) {
        place_in_cpu = false;
        break;
      }

      // input is a CPU tensor, but it's intended to be consumed as CPU input by the target EP
      if (node_to_kernel[cur]->kernel_def->IsInputOnCpu(i)) {
        place_in_cpu = false;
        break;
      }
    }

    if (place_in_cpu) {
      cpu_nodes.insert(cur);
      LOGS_DEFAULT(VERBOSE) << "ORT optimization- Force fallback to CPU execution for node: " << node->Name()
                            << " because the CPU execution path is deemed faster than overhead involved with execution on other EPs "
                            << " capable of executing this node";
      for (auto* output : node->OutputDefs()) {
        cpu_output_args.insert(output);
      }
      for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
        candidates.push((*it).Index());
      }
    }
  }

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)
  if (aggressive_cpu_fallback) {
    auto shape_related_node_indices = GetShapeRelatedNodes(graph);
    cpu_nodes.insert(shape_related_node_indices.begin(), shape_related_node_indices.end());
  }
#endif
  return cpu_nodes;
}

}  // namespace onnxruntime
