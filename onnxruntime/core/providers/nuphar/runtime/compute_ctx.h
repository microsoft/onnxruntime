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
#include "gsl/gsl"

#include <functional>
#include <mutex>
#include <vector>

// TODO change name space from tvm_codegen to nuphar
namespace onnxruntime {
namespace nuphar {

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

  void CreateFuncComputeCtx(const NupharFuncInfo* func_info, bool with_update = true);

  inline void UpdateFuncComputeCtx(const NupharFuncInfo* func_info) {
    ORT_ENFORCE_DEBUG(nullptr != func_info);

    FuncComputeCtx& func_compute_ctx = func_compute_ctx_map_.at(func_info);

    const auto& ort_input_allocators = func_info->ort_input_allocators;
    const std::vector<bool>& ort_input_allocator_is_collided_output = func_info->ort_input_allocator_is_collided_output;
    std::vector<const void*>& ort_input_data = func_compute_ctx.ort_input_data;
    std::vector<const int64_t*>& ort_input_shapes = func_compute_ctx.ort_input_shapes;

    for (int i = 0; i < gsl::narrow_cast<int>(ort_input_allocators.size()); ++i) {
      int offset = ort_input_allocators[i].index;
      bool is_external = ort_input_allocators[i].is_external;
      if (is_external) {
        bool is_collided = ort_input_allocator_is_collided_output[i];
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
    const auto& ort_output_allocator = func_info->ort_output_allocators[index];

    int offset = ort_output_allocator.index;
    bool is_external = ort_output_allocator.is_external;

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

  inline bool IsInitialized(const NupharFuncInfo* func_info) const {
    ORT_ENFORCE_DEBUG(nullptr != func_info);
    return func_compute_ctx_map_.count(func_info) > 0;
  }

  // UpdateRealizedDims is used to sync realize dim
  // Note insert_inclusive_axis is introduced to adjusted shape.
  // It is commonly used in Scan or other subgraphs
  // when Tensors' shapes in a subgraph are sliced from the main grahp.
  // Using the sliced axis as insert_inclusive_axis can find the correct shape dim in the main graph
  inline void UpdateRealizedDims(
      const std::vector<std::pair<size_t, std::string>>& symbols,
      const int64_t* input_shape,
      size_t insert_inclusive_axis = 65535 /*minimal maximum of size_t*/) {
    for (const auto& s_pair : symbols) {
      size_t dim = s_pair.first;
      size_t adjusted_dim = dim;
      if (dim >= insert_inclusive_axis) {
        adjusted_dim = dim + 1;
      }

      int64_t dim_size = input_shape[adjusted_dim];
      const std::string& dim_param = s_pair.second;
      auto dim_value_iter = realized_dims_.find(dim_param);

      if (dim_value_iter == realized_dims_.end()) {
        std::lock_guard<std::mutex> lock(mutex_);
        realized_dims_.insert(std::make_pair(dim_param, dim_size));  // update new symbol
      } else if (dim_value_iter->second == Dimension_Unknown) {
        std::lock_guard<std::mutex> lock(mutex_);
        dim_value_iter->second = dim_size;  // update for a symbol
      } else {
        std::lock_guard<std::mutex> lock(mutex_);
        // a runtime error
        ORT_ENFORCE(dim_value_iter->second == dim_size,
                    "Input shape's symbolic dim mismatch.", dim_value_iter->second, "!=", dim_size);
      }
    }
  }

  // UpdateRealizedDims is used to sync realize dim
  // Note insert_inclusive_axis is introduced to adjusted shape.
  // It is commonly used in Scan or other subgraphs
  // when Tensors' shapes in a subgraph are sliced from the main grahp.
  // Using the sliced axis as insert_inclusive_axis can find the correct shape dim in the main graph
  inline void UpdateRealizedDims(
      const std::vector<std::pair<size_t, std::string>>& symbols,
      std::vector<int64_t>& realized_output_shape,
      size_t insert_inclusive_axis = 65535 /*minimal maximum of size_t*/) {
    for (const auto& s_pair : symbols) {
      size_t dim = s_pair.first;
      size_t adjusted_dim = dim;
      if (dim >= insert_inclusive_axis) {
        adjusted_dim = dim + 1;
      }

      const std::string& dim_param = s_pair.second;
      auto dim_value_iter = realized_dims_.find(dim_param);
      ORT_ENFORCE_DEBUG(dim_value_iter != realized_dims_.end());
      {
        std::lock_guard<std::mutex> lock(mutex_);
        realized_output_shape[adjusted_dim] = dim_value_iter->second;
      }
    }
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

  std::mutex mutex_;
};

}  // namespace nuphar
}  // namespace onnxruntime
