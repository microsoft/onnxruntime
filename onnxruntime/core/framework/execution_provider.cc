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
}

common::Status IExecutionProvider::Sync() const { return Status::OK(); };

common::Status IExecutionProvider::OnRunStart() { return Status::OK(); }

common::Status IExecutionProvider::OnRunEnd() { return Status::OK(); }

common::Status IExecutionProvider::OnSessionInitializationEnd() { return Status::OK(); }

std::vector<AllocatorPtr> IExecutionProvider::GetAllocators() {
  std::vector<AllocatorPtr> retval;
  retval.reserve(allocators_.size());
  for (const auto& val : allocators_) {
    retval.push_back(val.second);
  }
  return retval;
}

void IExecutionProvider::InsertAllocatorHelper(AllocatorPtr allocator, bool allow_overwrite) {
  const OrtMemoryInfo& info = allocator->Info();
  auto ite = allocator_set_.find(info);
  bool already_added = ite != allocator_set_.end();
  if (!allow_overwrite && already_added) {
    ORT_THROW("duplicated allocator");
  }
  const int key = MakeKey(info.id, info.mem_type);
  allocators_[key] = {key, allocator};
  if (!already_added) {
    allocator_set_.insert(ite, info);
  }
}

void IExecutionProvider::InsertAllocator(AllocatorPtr allocator, bool allow_overwrite) {
  InsertAllocatorHelper(allocator, allow_overwrite);
}

void IExecutionProvider::InsertAllocator(AllocatorPtr allocator) {
  InsertAllocatorHelper(allocator, false);
}

common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_node*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}

common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_node*/,
                                           std::string& /*dll_path*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}

std::shared_ptr<KernelRegistry> IExecutionProvider::GetKernelRegistry() const {
  return nullptr;
}

}  // namespace onnxruntime
