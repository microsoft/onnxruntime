// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "internal_testing_execution_provider.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

constexpr const char* INTERNAL_TESTING_EP = "InternalTestingEP";

InternalTestingExecutionProvider::InternalTestingExecutionProvider(const std::unordered_set<std::string>& ops)
    : IExecutionProvider{utils::kInternalTestingExecutionProvider, true},
      ops_{ops} {
  // TODO: Allocation planner calls GetAllocator for the individual EP. It would be better if it goes through
  // the session state to get the allocator so it's per-device (or for the allocation planner to try the EP first
  // and fall back to using session state next by passing in a functor it can use to call SessionState::GetAllocator).

  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(INTERNAL_TESTING_EP,
                                                                    OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

InternalTestingExecutionProvider::~InternalTestingExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
InternalTestingExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  /* 
  Very basic search for groups of nodes that can be handled by the EP.
  This doesn't work perfectly if you have a scenario like the following where A and D could be handled by the EP
  but B is between them in the topological sort as you'll get two single node capabilities. However if can also
  be advantageous if C and E could be handled by the EP as they would be combined with D even though not connected.
  Not sure how often each of these scenarios happens. 

    A  B  C
    | /   |
    D     E
    |     |
    
  Would probably be better to walk the edges for each node the EP can handle as they are iterated in topological order,
  accumulating nodes (and saving which ones have been taken) until you run out. This would guarantee all
  connected nodes that can be handled are grouped together.     
  */

  std::vector<std::vector<NodeIndex>> node_groups;
  std::vector<NodeIndex> cur_group;
  for (NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    if (ops_.count(graph_viewer.GetNode(node_index)->OpType())) {
      cur_group.push_back(node_index);
    } else if (!cur_group.empty()) {
      node_groups.push_back(std::move(cur_group));
    }
  }

  if (!cur_group.empty()) {
    node_groups.push_back(std::move(cur_group));
  }

  if (node_groups.empty()) {
    return result;
  }

  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  for (const auto& group : node_groups) {
    std::unordered_set<NodeIndex> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(index);
    }

    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

    std::unordered_set<const NodeArg*> node_outputs;
    std::unordered_set<const NodeArg*> subgraph_inputs;
    std::unordered_set<const NodeArg*> subgraph_outputs;
    std::vector<const NodeArg*> ordered_subgraph_inputs;
    std::vector<const NodeArg*> ordered_subgraph_outputs;

    for (const auto& index : group) {
      sub_graph->nodes.push_back(index);
      const auto* node = graph_viewer.GetNode(index);

      for (const auto* input : node->InputDefs()) {
        // if the node input was not produced by this subgraph, add it to the subgraph inputs.
        if (node_outputs.count(input) == 0) {
          if (subgraph_inputs.count(input) == 0) {
            subgraph_inputs.insert(input);
            ordered_subgraph_inputs.push_back(input);
          }
        }
      }

      const auto& output_defs = node->OutputDefs();
      for (const auto* output_def : output_defs) {
        node_outputs.insert(output_def);
        // if output is overall graph output we need to produce it.
        if (graph_outputs.count(output_def) != 0) {
          ordered_subgraph_outputs.push_back(output_def);
        }
      }

      // if output connects to a node not in this subgraph we need to produce it
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        if (node_set.count(it->GetNode().Index()) == 0) {
          const auto* output_def = output_defs[it->GetSrcArgIndex()];
          if (subgraph_outputs.count(output_def) == 0) {
            subgraph_outputs.insert(output_def);
            ordered_subgraph_outputs.push_back(output_def);
          }
        }
      }
    }

    // Assign inputs and outputs to subgraph's meta_def
    uint64_t model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    meta_def->name = "InternalTestingEP_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    meta_def->domain = "InternalTesting";
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

    for (const auto& input : ordered_subgraph_inputs) {
      meta_def->inputs.push_back(input->Name());
    }

    for (const auto& output : ordered_subgraph_outputs) {
      meta_def->outputs.push_back(output->Name());
    }

    sub_graph->SetMetaDef(std::move(meta_def));

    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  return result;
}

common::Status InternalTestingExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                                                         std::vector<NodeComputeInfo>& node_compute_funcs) {
  // Create a function to generate dummy empty output for each fused node so the model can be executed.
  for (const auto& node_and_viewer : fused_nodes) {
    NodeComputeInfo compute_info;
    const Node& node = node_and_viewer.fused_node;

    //{
    //  const GraphViewer& graph_viewer = node_and_viewer.filtered_graph;
    //  std::cout << "Fusing nodes: ";
    //  for (const auto& unfused_node : graph_viewer.Nodes()) {
    //    std::cout << " '" << unfused_node.Name() << "':" << unfused_node.Index();
    //  }
    //  std::cout << std::endl;
    //}

    compute_info.create_state_func = [](ComputeContext* /*context*/, FunctionState* /*state*/) {
      return 0;
    };

    compute_info.release_state_func = [](FunctionState /*state*/) {
    };

    compute_info.compute_func = [&node](FunctionState /*state*/, const OrtCustomOpApi* c_api,
                                        OrtKernelContext* context) -> Status {
      Ort::CustomOpApi api{*c_api};  // use C++ API for convenience

      const auto outputs = node.OutputDefs();
      const size_t num_outputs = outputs.size();

      for (size_t i = 0; i < num_outputs; i++) {
        const auto* shape_proto = outputs[i]->Shape();
        if (shape_proto == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown output shapes are not supported");
        }

        TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
        if (shape.Size() < 0) {
          // arbitrarily set any unknown dim to 1
          for (size_t idx = 0, end = shape.NumDimensions(); idx < end; ++idx) {
            if (shape[idx] == -1) {
              shape[idx] = 1;
            }
          }
        }

        // create the output_tensor.
        auto* ortvalue = api.KernelContext_GetOutput(context, i, shape.GetDims().data(), shape.GetDims().size());

        // and fill with zeros
        auto* tensor = ortvalue->GetMutable<Tensor>();
        void* data = tensor->MutableDataRaw();
        auto bytes = tensor->SizeInBytes();
        memset(data, 0, bytes);
      };

      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}
}  // namespace onnxruntime
