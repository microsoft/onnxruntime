// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/macros.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/data_types.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/callback.h"
#include "core/optimizer/optimizer_execution_frame.h"

namespace onnxruntime {

OptimizerExecutionFrame::Info::Info(const std::vector<const Node*>& nodes,
                                    const InitializedTensorSet& initialized_tensor_set) {
  // Create CPU execution provider
  // For now, CPU execution provider will be created every time when initializing Info.
  // Later, it will be changed to pass by Info ctor.
  cpu_execution_provider_ = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  allocator_ptr_ = cpu_execution_provider_->GetAllocator(device_id_, mem_type_);
  ORT_ENFORCE(allocator_ptr_ != nullptr, "Failed to get allocator for optimizer");

  data_transfer_mgr_.RegisterDataTransfer(onnxruntime::make_unique<CPUDataTransfer>());

  // Create MLValues related maps
  auto initialize_maps = [this, &initialized_tensor_set](const NodeArg& arg, size_t /*index*/) -> Status {
    int idx = ort_value_name_idx_map_.Add(arg.Name());
    ort_value_idx_nodearg_map_[idx] = &arg;

    // Only create OrtValue instances for initializers used by an array of nodes.
    InitializedTensorSet::const_iterator it = initialized_tensor_set.find(arg.Name());
    if (it != initialized_tensor_set.cend()) {
      const auto& tensor_proto = *(it->second);
      size_t cpu_tensor_length;
      ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(tensor_proto, &cpu_tensor_length));
      OrtValue ort_value;
      const OrtMemoryInfo& info = cpu_execution_provider_->GetAllocator(0, OrtMemTypeDefault)->Info();
      std::unique_ptr<char[]> data(new char[cpu_tensor_length]);
      std::unique_ptr<Tensor> p_tensor;
      OrtCallback d;
      ORT_RETURN_IF_ERROR(utils::TensorProtoToMLValue(Env::Default(), nullptr, tensor_proto,
                                                      MemBuffer(data.get(), cpu_tensor_length, info), ort_value, d));

      initializers_[idx] = ort_value;
      buffer_for_initialized_tensors_[idx] = std::move(data);
      if (d.f != nullptr)
        deleter_for_initialized_tensors_[idx] = d;
    }

    return Status::OK();
  };

  // TODO: node->ImplicitInputDefs() need to be added here for control flow nodes.
  for (auto* node : nodes) {
    onnxruntime::Node::ForEachWithIndex(node->InputDefs(), initialize_maps);
    onnxruntime::Node::ForEachWithIndex(node->OutputDefs(), initialize_maps);
  }

  node_index_info_ = onnxruntime::make_unique<NodeIndexInfo>(nodes, ort_value_name_idx_map_);

  // create kernels for these nodes
  for (auto* node : nodes) {
    std::unique_ptr<OpKernel> op_kernel;
    std::shared_ptr<KernelRegistry> kernel_registry = cpu_execution_provider_->GetKernelRegistry();
    auto status = kernel_registry->TryCreateKernel(*node, *cpu_execution_provider_, initializers_,
                                                   ort_value_name_idx_map_, FuncManager(), data_transfer_mgr_, op_kernel);
    kernels_[node->Index()] = std::move(op_kernel);
  }
}

const OpKernel* OptimizerExecutionFrame::Info::GetKernel(NodeIndex node_id) const {
  if (kernels_.find(node_id) == kernels_.cend()) {
    return nullptr;
  }

  return kernels_.at(node_id).get();
}

// For optimizer, probably no need to pass feed_mlvalue_idxs, feeds to initialize IExecutionFrame.
// If needed, the parameters of OptimizerExecutionFrame ctor can be changed later.
OptimizerExecutionFrame::OptimizerExecutionFrame(const Info& info, const std::vector<int>& fetch_mlvalue_idxs)
    : IExecutionFrame(std::vector<int>(), std::vector<OrtValue>(), info.GetInitializers(), fetch_mlvalue_idxs,
                      std::vector<OrtValue>(), info.GetMLValueNameIdxMap(), info.GetNodeIndexInfo()),
      info_(info) {}

AllocatorPtr OptimizerExecutionFrame::GetAllocatorImpl(const OrtMemoryInfo& info) const {
  return info_.GetAllocator(info);
}

// This method is not thread safe!
// Return S_OK and nullptr if index map to an value that is an unused optional input/output
Status OptimizerExecutionFrame::CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_idx,
                                                            const TensorShape* shape, size_t nnz) {
  const DataTypeImpl* ml_type = utils::GetMLDataType(*(info_.GetMLValueIdxNodeArgMap().at(ort_value_idx)));
  if (ml_type == nullptr)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Tried to allocate without valid type information, ort_value index=" + std::to_string(ort_value_idx));
  if (ml_type->IsSparseTensorType()) {
    auto element_type = ml_type->AsSparseTensorType()->GetElementType();
    auto container_type = DataTypeImpl::GetType<SparseTensor>();
    auto sparse = onnxruntime::make_unique<SparseTensor>(element_type, *shape, nnz, info_.GetAllocator());
    ort_value.Init(sparse.release(), container_type, container_type->GetDeleteFunc());
    return Status::OK();
  }
  if (!ml_type->IsTensorType()) {
    const NonTensorTypeBase* non_tensor_type = static_cast<const NonTensorTypeBase*>(ml_type);
    auto creator = non_tensor_type->GetCreateFunc();
    ort_value.Init(creator(), non_tensor_type, non_tensor_type->GetDeleteFunc());
    return Status::OK();
  }

  // tensors
  auto element_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
  AllocatorPtr allocator_ptr = info_.GetAllocator();
  std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                              *shape,
                                                              allocator_ptr);

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());

  return Status::OK();
}

}  // namespace onnxruntime
