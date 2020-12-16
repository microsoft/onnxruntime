// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/execution_provider.h"

#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

namespace {
//It assumes max(OrtMemType) <= 1, min(OrtMemType) = -2
inline int MakeKey(int id, OrtMemType mem_type) {
  return id << 2 | (mem_type + 2);
}
}  // namespace

AllocatorPtr IExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  auto iter = allocators_.find(MakeKey(id, mem_type));
  if (iter != allocators_.end()) {
    return iter->second;
  }
  return nullptr;
}

std::vector<std::unique_ptr<ComputeCapability>>
IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                  const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
#if !defined(ORT_MINIMAL_BUILD)
  for (auto& node : graph.Nodes()) {
    for (auto registry : kernel_registries) {
      if (KernelRegistry::HasImplementationOf(*registry, node, Type())) {
        std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(node.Index());
        result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
        break;
      }
    }
  }

  return result;
#else
  // We have saved hashes to lookup static kernels in an ORT format model so the default behavior is to return an
  // empty vector to leave that in place. An EP that compiles nodes can override this in a minimal build.
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(kernel_registries);
  return result;
#endif
}

// Update allocator in the provider if already present; ignore if not.
void IExecutionProvider::ReplaceAllocator(AllocatorPtr allocator) {
  const auto& info = allocator->Info();
  auto ite = mem_info_set_.find(info);
  if (ite != mem_info_set_.end()) {
    const int key = MakeKey(info.id, info.mem_type);
    allocators_[key] = allocator;
  }
}

void IExecutionProvider::InsertAllocator(AllocatorPtr allocator) {
  const OrtMemoryInfo& info = allocator->Info();
  auto ite = mem_info_set_.find(info);
  if (ite != mem_info_set_.end()) {
    ORT_THROW("duplicated allocator");
  }
  const int key = MakeKey(info.id, info.mem_type);
  allocators_.insert({key, allocator});
  mem_info_set_.insert(ite, info);
  allocator_list_.push_back(allocator);
}

#if !defined(ORT_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_node*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}

common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_node*/,
                                           std::string& /*dll_path*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& /*fused_nodes_and_graphs*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}
#endif

}  // namespace onnxruntime
