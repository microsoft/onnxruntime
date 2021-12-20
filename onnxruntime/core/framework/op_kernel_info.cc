// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {

OpKernelInfo::OpKernelInfo(const onnxruntime::Node& node,
                           const KernelDef& kernel_def,
                           const IExecutionProvider& execution_provider,
                           const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                           const OrtValueNameIdxMap& ort_value_name_idx_map,
                           const DataTransferManager& data_transfer_mgr)
    : OpNodeProtoHelper(&proto_helper_context_),
      node_(node),
      kernel_def_(kernel_def),
      execution_provider_(&execution_provider),
      constant_initialized_tensors_(constant_initialized_tensors),
      ort_value_name_idx_map_(ort_value_name_idx_map),
      data_transfer_mgr_(data_transfer_mgr),
      proto_helper_context_(node) {}

OpKernelInfo::OpKernelInfo(const OpKernelInfo& other)
    : OpKernelInfo(other.node_, other.kernel_def_, *other.execution_provider_, other.constant_initialized_tensors_,
                   other.ort_value_name_idx_map_, other.data_transfer_mgr_) {}

const OrtMemoryInfo& OpKernelInfo::GetMemoryInfo(int device_id, OrtMemType mem_type) const {
  AllocatorPtr alloc = GetAllocator(device_id, mem_type);
  if (alloc == nullptr) ORT_THROW("cannot find allocator");
  return alloc->Info();
}

AllocatorPtr OpKernelInfo::GetAllocator(int device_id, OrtMemType mem_type) const {
  return execution_provider_->GetAllocator(device_id, mem_type);
}

const KernelDef& OpKernelInfo::GetKernelDef() const {
  return kernel_def_;
}

const IExecutionProvider* OpKernelInfo::GetExecutionProvider() const noexcept {
  return execution_provider_;
}

const DataTransferManager& OpKernelInfo::GetDataTransferManager() const noexcept {
  return data_transfer_mgr_;
}

const onnxruntime::Node& OpKernelInfo::node() const noexcept {
  return node_;
}

bool OpKernelInfo::TryGetConstantInput(int input_index, const Tensor** constant_input_value) const {
  if (input_index < 0 || input_index >= gsl::narrow_cast<int>(node_.InputDefs().size())) {
    return false;
  }
  auto& input_arg_name = node_.InputDefs()[input_index]->Name();
  int input_arg_index = -1;
  if (!ort_value_name_idx_map_.GetIdx(input_arg_name, input_arg_index).IsOK()) {
    return false;
  }

  auto iter = constant_initialized_tensors_.find(input_arg_index);
  if (constant_initialized_tensors_.end() == iter) {
    return false;
  }

  if (!iter->second.IsTensor()) {
    // Only constant Tensor input is supported right now, since we're using initializers to store the data.
    return false;
  }

  *constant_input_value = &iter->second.Get<Tensor>();
  return true;
}

}  // namespace onnxruntime
