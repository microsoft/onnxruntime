// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/compute_ctx.h"

namespace onnxruntime {
namespace tvm_codegen {

KernelComputeCtx::KernelComputeCtx(
    const nuphar::NupharRuntimeHandle* handle,
    std::unordered_map<std::string, int64_t>& realized_dims,
    DataAllocFunc data_alloc_func,
    int allocator_offset_count)
    : data_alloc_func_(data_alloc_func),
      handle_(handle),
      realized_dims_(realized_dims) {
  internal_ort_buffer_unique_ptrs_.resize(allocator_offset_count);
}

void KernelComputeCtx::CreateFuncComputeCtx(const NupharFuncInfo* func_info) {
  ORT_ENFORCE_DEBUG(nullptr != func_info);

  if (func_compute_ctx_map_.find(func_info) == func_compute_ctx_map_.end()) {
    func_compute_ctx_map_.emplace(func_info, FuncComputeCtx());
  }

  FuncComputeCtx& func_compute_ctx = func_compute_ctx_map_.at(func_info);
  const std::vector<int>& ort_input_to_allocator_indices = func_info->ort_input_to_allocator_indices;
  size_t num_input = ort_input_to_allocator_indices.size();
  std::vector<const void*>& ort_input_data = func_compute_ctx.ort_input_data;
  std::vector<const int64_t*>& ort_input_shapes = func_compute_ctx.ort_input_shapes;
  ort_input_data.resize(num_input);
  ort_input_shapes.resize(num_input);

  UpdateFuncComputeCtx(func_info);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
