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
#include "core/framework/func_kernel.h"
#include "core/framework/session_state.h"

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

Status GraphPartitioner::Partition(onnxruntime::Graph& graph, const SessionState& session_state) const {
  if (providers_.Empty()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No provider specified.");
  }
  //fused_kernel_registry is prepareing the kernels created on the fly for fused sub graph.
  //It is only visiable for current session.
  std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();
  // Partitioning <graph> based on provider preference and their capabilities.
  auto kernel_registries = kernel_registry_mgr_.GetAllKernelRegistries();
  for (auto& provider : providers_) {
    auto capability_results = provider->GetCapability(GraphViewer(graph), kernel_registries);
    int count = 0;
    std::vector<Node*> nodes_need_compile;
    for (auto& capability : capability_results) {
      if (nullptr == capability || nullptr == capability->sub_graph) {
        continue;
      }
      if (nullptr == capability->sub_graph->GetMetaDef()) {
        // The <provider> can run a single node in the <graph> if not using meta-defs.
        // A fused kernel is not supported in this case.
        ORT_ENFORCE(1 == capability->sub_graph->nodes.size());

        auto node = graph.GetNode(capability->sub_graph->nodes[0]);
        if (nullptr != node && node->GetExecutionProviderType().empty()) {
          node->SetExecutionProviderType(provider->Type());
        }
      } else {
        // The <provider> can run a fused <sub_graph> in the <graph>.
        //
        // Add fused node into <graph>
        ORT_ENFORCE(nullptr != capability->sub_graph->GetMetaDef());
        std::string node_name = provider->Type() + "_" + capability->sub_graph->GetMetaDef()->name + "_" + std::to_string(count++);
        auto& fused_node = graph.FuseSubGraph(std::move(capability->sub_graph), node_name);
        fused_node.SetExecutionProviderType(provider->Type());
        if (capability->need_compile)
          nodes_need_compile.push_back(&fused_node);
      }
    }
    if (nodes_need_compile.size() > 0) {
      if (session_state.ExportDll()) {
        std::string dll_path;
        ORT_RETURN_IF_ERROR(provider->Compile(nodes_need_compile, dll_path));
        for (auto* node : nodes_need_compile)
          ORT_RETURN_IF_ERROR(const_cast<FuseFuncManager*>(session_state.GetFusedFuncMgr())->AddFuncInfo(node->Name(), dll_path));
      } else {
        std::vector<NodeComputeInfo> node_compute_funcs;
        ORT_RETURN_IF_ERROR(provider->Compile(nodes_need_compile, node_compute_funcs));
        ORT_ENFORCE(node_compute_funcs.size() == nodes_need_compile.size(), "Provider doesn't return correct number of compiled functions");
        for (auto i = 0; i < nodes_need_compile.size(); i++)
          ORT_RETURN_IF_ERROR(const_cast<FuseFuncManager*>(session_state.GetFusedFuncMgr())->AddFuncInfo(nodes_need_compile[i]->Name(), node_compute_funcs[i].compute_func, node_compute_funcs[i].create_state_func, node_compute_funcs[i].release_state_func));
      }
      for (auto* node : nodes_need_compile) {
        //prepare the func kernel
        KernelDefBuilder builder;
        BuildFusedKernelDef(builder, *node);
        fused_kernel_registry->Register(builder, [](const OpKernelInfo& info) { return new FunctionKernel(info); });
      }
    }
    // all done with this provider, resolve the graph before we move on to the next provider.
    // This is needed since we create a new GraphViewer() that we pass into the next provider's GetCapability().
    ORT_ENFORCE(graph.Resolve().IsOK());
  }

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
    this->Partition(graph, session_state);
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
