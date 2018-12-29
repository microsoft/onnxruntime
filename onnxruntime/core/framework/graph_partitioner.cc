// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/graph_partitioner.h"

#include "core/framework/kernel_registry_manager.h"
#include "core/graph/function.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry.h"

// uncomment this line to count non-CUDA ops in ONNX domain
//#define COUNT_NON_CUDA_OPS

#ifdef COUNT_NON_CUDA_OPS
class NonCudaOps {
 public:
  ~NonCudaOps() {
    printf("Non-CUDA ops:\n");
    for (auto i : map_) {
      printf("%s: %d\n", i.first.c_str(), i.second);
    }
  }

  void AddOp(const std::string& name) {
    if (map_.count(name))
      map_.at(name)++;
    else
      map_.insert({name, 1});
  }

 private:
  std::map<std::string, int> map_;
};

NonCudaOps non_cuda;
#endif

using namespace ::onnxruntime::common;
namespace onnxruntime {

KernelDefBuilder& BuildFusedKernelDef(KernelDefBuilder& builder, const onnxruntime::Node& node) {
  auto schema = node.Op();
  builder.SetName(schema->Name())
      .SetDomain(schema->domain())
      .SinceVersion(schema->SinceVersion())
      .Provider(node.GetExecutionProviderType());
  auto& inputs = node.InputDefs();
  for (auto input : inputs) {
    builder.TypeConstraint(input->Name(), DataTypeImpl::TypeFromProto(*input->TypeAsProto()));
  }
  return builder;
}

Status GraphPartitioner::Partition(onnxruntime::Graph& graph) const {
  // It is a greedy partitioning algorithm per provider preferences user provided when calling ONNX RUNTIME right now.
  // 1. Execution providers' capabilities are checked one by one.
  // 2. All sub-graphs that an execution provider returns will be assigned to it if it's not assigned yet.
  // 3. CPU execution provider is expected to be able to run any node and is the last one in execution provider preference.

  if (providers_.Empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No provider specified.");
  }
  //fused_kernel_registry is prepareing the kernels created on the fly for fused sub graph.
  //It is only visiable for current session.
  std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();
  // Partitioning <graph> based on provider preference and their capabilities.
  auto kernel_registries = kernel_registry_mgr_.GetAllKernelRegistries();

  std::vector<std::vector<std::unique_ptr<ComputeCapability>>> capabilities_of_all_providers;
  GraphViewer graph_viewer(graph);
  for (auto& provider : providers_) {
    capabilities_of_all_providers.push_back(provider->GetCapability(graph_viewer, kernel_registries));
  }

  int i = 0;
  for (auto& provider : providers_) {
    int count = 0;
    for (auto& capability : capabilities_of_all_providers[i++]) {
      if (nullptr == capability || nullptr == capability->sub_graph) {
        continue;
      }
      if (nullptr == capability->sub_graph->GetMetaDef()) {
        // The <provider> can run a single node in the <graph> if not using meta-defs.
        // A fused kernel is not supported in this case.
        ORT_ENFORCE(1 == capability->sub_graph->nodes.size());
        ORT_ENFORCE(capability->fuse_kernel_function == nullptr);

        auto node = graph.GetNode(capability->sub_graph->nodes[0]);
        if (nullptr != node && node->GetExecutionProviderType().empty()) {
          // The node was not fused or assigned. Assign it to this <provider>.
          node->SetExecutionProviderType(provider->Type());
        }
      } else {
        // The <provider> can run a fused <sub_graph> in the <graph>.
        ORT_ENFORCE(nullptr != capability->sub_graph->GetMetaDef());

        // Check whether any node in the <sub_graph> was already assigned.
        bool sub_graph_available_for_assignment = true;
        for (auto node_index : capability->sub_graph->nodes) {
          auto node = graph.GetNode(node_index);
          if (nullptr == node || !node->GetExecutionProviderType().empty()) {
            // The node was fused or assigned, so that the whole sub-graph will not be assigned to this <provider>
            // The assumption is that this <provider> can only run the sub-graph as a whole unit.
            sub_graph_available_for_assignment = false;
            break;
          }
        }

        if (sub_graph_available_for_assignment) {
          // Add fused node into <graph>
          std::string node_name = provider->Type() + "_" + capability->sub_graph->GetMetaDef()->name + "_" + std::to_string(count++);
          auto& fused_node = graph.FuseSubGraph(std::move(capability->sub_graph), node_name);
          fused_node.SetExecutionProviderType(provider->Type());
          auto fused_kernel_func = capability->fuse_kernel_function;
          if (fused_kernel_func != nullptr) {
            // build the kernel definition on the fly, and register it to the fused_kernel_regisitry.
            KernelDefBuilder builder;
            BuildFusedKernelDef(builder, fused_node);
            fused_kernel_registry->Register(builder, fused_kernel_func);
          }
        }
      }
    }
  }

  ORT_ENFORCE(graph.Resolve().IsOK());

  // To see if the node with no provider can be inlined. If one such nodes can be
  // successfully inlined, we re-run the partitioner on the modified graph.
  bool inline_flag = false;
  for (auto& node : graph.Nodes()) {
    if (node.GetExecutionProviderType().empty()) {
      auto node_func = node.GetFunctionBody();
      if (nullptr == node_func) {
        continue;
      }
      Status inliner_status = graph.InlineFunction(node);
      // If the node has a functionbody with no kernel and cannot be inlined
      // it is a invalid function
      if (!inliner_status.IsOK()) return inliner_status;
      // Set the flag for re-run graph partition after successful inlining
      inline_flag = true;
      break;
    }
  }
  // Resolve and rerun graph partition
  if (inline_flag) {
    ORT_RETURN_IF_ERROR(graph.Resolve());
    this->Partition(graph);
  }

    //For some cases, like fp16 on cpu, right now we don't have any kernel support that.
    //But we will insert cast op to run the model, so skip the error checking here.
    //If after graph transform phase, the node still not assigned, we will report error
    //during kernel creation phase.
#ifdef COUNT_NON_CUDA_OPS
  for (auto& node : graph.Nodes()) {
    if (node.GetExecutionProviderType() != kCudaExecutionProvider &&
        node.Domain() != kMLDomain &&
        node.Domain() != kMSDomain)
      non_cuda.AddOp(node.OpType());
  }
#endif

  kernel_registry_mgr_.RegisterKernelRegistry(fused_kernel_registry, KernelRegistryPriority::HighPriority);

  return Status::OK();
}
}  // namespace onnxruntime
