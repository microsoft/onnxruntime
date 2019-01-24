
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/macros.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/mlvalue_name_idx_map.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/graph_optimizer.h"

namespace onnxruntime {

Status GraphOptimizer::Init() {
  GraphViewer viewer(graph_);

  // Create a CPU execution provider
  LOGS(logger_, INFO) << "Adding default CPU execution provider.";
  CPUExecutionProviderInfo info;
  //std::unique_ptr<CPUExecutionProvider> cpu_execution_provider = std::make_unique<CPUExecutionProvider>(info);
  execution_providers_.Add(onnxruntime::kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(info));
  auto* cpu_execution_provider = execution_providers_.Get(onnxruntime::kCpuExecutionProvider);

  // Create MLValueNameIdxMap
  MLValueNameIdxMap& mlvalue_name_idx_map = session_state_.GetMLValueNameIdxMap();
  for (const auto& pair : graph_.GetAllInitializedTensors()) {
    const auto& initializer_name = pair.first;
    int index = mlvalue_name_idx_map.Add(initializer_name);
  }
  ORT_ENFORCE(mlvalue_name_idx_map.MaxIdx() > 0, "MLValue indexes should have been populated.");

  // De-serialize tensors
  LOGS(logger_, INFO) << "Saving initialized tensors.";
  const onnxruntime::InitializedTensorSet& initialized_tensor_set = graph_.GetAllInitializedTensors();
  for (const auto& entry : initialized_tensor_set) {
    const std::string& name = entry.first;
    int mlvalue_index;
    ORT_RETURN_IF_ERROR(mlvalue_name_idx_map.GetIdx(name, mlvalue_index));
    VLOGS(logger_, 1) << "About to add weight with name: " << name << " and index: " << mlvalue_index;

    // CPU
    OrtAllocatorInfo allocator_info = cpu_execution_provider->GetAllocator(0, OrtMemTypeDefault)->Info();
    auto allocator_ptr = cpu_execution_provider->GetAllocator(allocator_info.id, allocator_info.mem_type);
    if (!allocator_ptr) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get allocator for alloc_info: " + allocator_info.ToString());
    }
    MLValue mlvalue;
    // deserialize directly to CPU tensor
    return utils::TensorProtoToMLValue(*(entry.second), allocator_ptr, nullptr, 0, mlvalue);

    // save mlvalue in a map
    //std::unordered_map<int, MLValue> initialized_tensors_;  // key is mlvalue_index
    //initialized_tensors_.insert({mlvalue_index, mlvalue});
    session_state_.AddInitializedTensor(mlvalue_index, mlvalue);
    VLOGS(logger_, 1) << "Added weight with name : " << name << " with index: " << mlvalue_index;
  }
  LOGS(logger_, INFO) << "Done saving initialized tensors";

  // Create MLValue for output of constant folding nodes.

  // Get CPU register kernels
  LOGS(logger_, INFO) << "Saving kernels.";
  for (auto& node : viewer.Nodes()) {
    // construct and save the kernels
    std::unique_ptr<OpKernel> op_kernel;
    std::shared_ptr<KernelRegistry> kernel_registry = cpu_execution_provider->GetKernelRegistry();

    auto status = kernel_registry->CreateKernel(node, *cpu_execution_provider, session_state_, op_kernel);
    session_state_.AddKernel(node.Index(), std::move(op_kernel));
  }
  LOGS(logger_, INFO) << "Done saving kernels.";

  return Status::OK();

}
}