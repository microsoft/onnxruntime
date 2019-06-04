// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nuphar/compiler/func_info.h"
#include "core/providers/nuphar/runtime/handle.h"
#include "core/providers/nuphar/runtime/control_flow/loop_exec_ctx.h"

#include "core/codegen/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/func_api.h"
#include "core/framework/func_kernel.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "gsl/gsl_util"

#include <functional>
#include <vector>

// TODO change name space from tvm_codegen to nuphar
namespace onnxruntime {
namespace tvm_codegen {

class LoopExecCtx;

using DataAllocFunc = std::function<void*(size_t)>;

struct FuncComputeCtx {
  // ort context
  std::vector<const void*> ort_input_data;
  std::vector<const int64_t*> ort_input_shapes;

  // tvm context
  std::vector<TVMValue> lvalues;
  std::vector<DLTensor> dl_tensors;
  std::vector<std::vector<int64_t>> dl_output_shapes;

  // LoopExecCtx
  std::unique_ptr<LoopExecCtx> loop_cf_ctx;
};

struct InternalTensor {
  IAllocatorUniquePtr<void> allocator_ptr;
  const int64_t* shape;
};

// KernelComputeCtx is a stateful data struct
class KernelComputeCtx {
 public:
  explicit KernelComputeCtx(
      const nuphar::NupharRuntimeHandle* handle,
      std::unordered_map<std::string, int64_t>& realized_dims,
      DataAllocFunc data_alloc_func,
      int allocator_offset_count);

  inline void Bind(OpKernelContext* op_kernel_ctx) {
    op_kernel_ctx_ = op_kernel_ctx;
  }

  void CreateFuncComputeCtx(const NupharFuncInfo* func_info);

  inline void UpdateFuncComputeCtx(const NupharFuncInfo* func_info) {
    ORT_ENFORCE_DEBUG(nullptr != func_info);

    FuncComputeCtx& func_compute_ctx = func_compute_ctx_map_.at(func_info);
    const std::vector<int>& ort_input_to_allocator_indices = func_info->ort_input_to_allocator_indices;
    const std::vector<bool>& ort_input_allocator_index_is_external = func_info->ort_input_allocator_index_is_external;
    const std::vector<bool>& ort_input_allocator_index_is_collided_output = func_info->ort_input_allocator_index_is_collided_output;
    std::vector<const void*>& ort_input_data = func_compute_ctx.ort_input_data;
    std::vector<const int64_t*>& ort_input_shapes = func_compute_ctx.ort_input_shapes;

    for (int i = 0; i < gsl::narrow_cast<int>(ort_input_to_allocator_indices.size()); ++i) {
      int offset = ort_input_to_allocator_indices[i];
      bool is_external = ort_input_allocator_index_is_external[i];
      if (is_external) {
        bool is_collided = ort_input_allocator_index_is_collided_output[i];
        if (is_collided) {
          const Tensor* t = op_kernel_ctx_->Output<Tensor>(offset);
          ort_input_data[i] = t->DataRaw();
          ort_input_shapes[i] = t->Shape().GetDims().data();
        } else {
          const Tensor* t = op_kernel_ctx_->Input<Tensor>(offset);
          ort_input_data[i] = t->DataRaw();
          ort_input_shapes[i] = t->Shape().GetDims().data();
        }
      } else {
        const InternalTensor& t = internal_ort_buffer_unique_ptrs_[offset];
        ort_input_data[i] = t.allocator_ptr.get();
        ort_input_shapes[i] = t.shape;
      }
    }
  }

  inline FuncComputeCtx& GetFuncComputeCtx(const NupharFuncInfo* func_info) {
    return func_compute_ctx_map_.at(func_info);
  }

  inline const FuncComputeCtx& GetFuncComputeCtx(const NupharFuncInfo* func_info) const {
    return func_compute_ctx_map_.at(func_info);
  }

  inline void* OutputData(const NupharFuncInfo* func_info,
                          int index,
                          const TensorShape& shape,
                          MLDataType dtype) {
    int offset = func_info->ort_output_to_allocator_indices[index];
    bool is_external = func_info->ort_output_allocator_index_is_external[index];

    if (is_external) {
      auto t = op_kernel_ctx_->Output(offset, shape);
      return t->MutableDataRaw();
    }

    internal_ort_buffer_unique_ptrs_[offset].allocator_ptr = AllocateDataUniquePtr(shape, dtype);
    internal_ort_buffer_unique_ptrs_[offset].shape = shape.GetDims().data();
    return internal_ort_buffer_unique_ptrs_[offset].allocator_ptr.get();
  }

  inline IAllocatorUniquePtr<void> AllocateDataUniquePtr(const int64_t* shape, const size_t rank, MLDataType dtype) {
    int64_t total_size = dtype->Size();
    for (size_t i = 0; i < rank; ++i) {
      total_size *= shape[i];
    }
    return IAllocator::MakeUniquePtr<void>(handle_->allocator, total_size);
  }

  inline const nuphar::NupharRuntimeHandle* GetRuntimeHandle() const {
    return handle_;
  }

  inline std::unordered_map<std::string, int64_t>& GetRealizedDims() {
    return realized_dims_;
  }

 private:
  inline IAllocatorUniquePtr<void> AllocateDataUniquePtr(const TensorShape& shape, MLDataType dtype) {
    return IAllocator::MakeUniquePtr<void>(handle_->allocator, shape.Size() * dtype->Size());
  }

  DataAllocFunc data_alloc_func_;

  // runtime handle
  const nuphar::NupharRuntimeHandle* handle_;

  // realized_dim
  std::unordered_map<std::string, int64_t>& realized_dims_;

  // ORT Kernel Context
  OpKernelContext* op_kernel_ctx_;

  // Owner of non-real inputs, outputs, and even state buffers
  // Here, "non-real" means edge across subgraph only, but not across boundary of Partition.
  // Non-real inputs and outputs is managed by Nuphar.
  std::vector<InternalTensor> internal_ort_buffer_unique_ptrs_;

  std::map<const NupharFuncInfo*, FuncComputeCtx> func_compute_ctx_map_;
};

}  // namespace tvm_codegen
}  // namespace onnxruntime
