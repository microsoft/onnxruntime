// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/execution_provider.h"

#include "core/graph/graph.h"
#include "core/framework/computation_capacity.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

namespace {
inline int MakeKey(int id, ONNXRuntimeMemType mem_type) {
  return id << 2 | mem_type;
}
}  // namespace

AllocatorPtr IExecutionProvider::GetAllocator(int id, ONNXRuntimeMemType mem_type) const {
  auto iter = allocators_.find(MakeKey(id, mem_type));
  if (iter != allocators_.end()) {
    return iter->second;
  }
  return nullptr;
}

std::vector<std::unique_ptr<ComputationCapacity>>
IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                  const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputationCapacity>> result;
  for (auto& node : graph.Nodes()) {
    for (auto registry : kernel_registries) {
      if (registry->TryFindKernel(node, Type()) != nullptr) {
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(node.Index());
        result.push_back(std::make_unique<ComputationCapacity>(std::move(sub_graph), nullptr));
      }
    }
  }

  return result;
}

common::Status IExecutionProvider::CopyTensor(const Tensor& src,
                                              Tensor& dst,
                                              int exec_queue_id) const {
  // execution provider may override this to support different exec queues
  ONNXRUNTIME_ENFORCE(exec_queue_id == 0);
  return CopyTensor(src, dst);
}

common::Status IExecutionProvider::Sync() const { return Status::OK(); };

common::Status IExecutionProvider::OnRunStart() { return Status::OK(); }

common::Status IExecutionProvider::OnRunEnd() { return Status::OK(); }

void IExecutionProvider::InsertAllocator(AllocatorPtr allocator) {
  const ONNXRuntimeAllocatorInfo& info = allocator->Info();
  const int key = MakeKey(info.id, info.mem_type);
  auto iter = allocators_.find(key);
  if (iter != allocators_.end()) {
    ONNXRUNTIME_THROW("duplicated allocator");
  }
  allocators_.insert(iter, {key, allocator});
}
}  // namespace onnxruntime
