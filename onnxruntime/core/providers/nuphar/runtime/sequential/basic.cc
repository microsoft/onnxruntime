// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/basic.h"

#include "core/codegen/common/common.h"
#include "core/codegen/common/profile.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "gsl/gsl"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

void BasicExecBlock::Run(KernelComputeCtx* kernel_compute_ctx) {
  CODEGEN_PROFILER_EVENT(func_info_->name);
  if (!kernel_compute_ctx->IsInitialized(func_info_)) {
    kernel_compute_ctx->CreateFuncComputeCtx(func_info_);
    InitContext(kernel_compute_ctx);
  } else {
    kernel_compute_ctx->UpdateFuncComputeCtx(func_info_);
    UpdateContext(kernel_compute_ctx);
  }

  FuncComputeCtx& subgraph_compute_ctx = kernel_compute_ctx->GetFuncComputeCtx(func_info_);
  size_t tvm_input_count = func_info_->func_input_count;
  size_t tvm_output_count = func_info_->func_output_count;
  int tvm_num_args = gsl::narrow<int>(tvm_input_count + tvm_output_count);
  tvm::TVMArgs tvm_args(subgraph_compute_ctx.lvalues.data(),
                        func_info_->type_codes.data(),
                        tvm_num_args);

  tvm::TVMRetValue rvalue;
  const tvm::runtime::PackedFunc& func = func_info_->packed_func;

  func.CallPacked(tvm_args, &rvalue);

  // Check aliased outputs
  if (tvm_output_count < func_info_->ort_output_count) {
    const std::vector<DLTensor>& dl_tensors = subgraph_compute_ctx.dl_tensors;
    const std::vector<std::vector<int64_t>>& dl_output_shapes = subgraph_compute_ctx.dl_output_shapes;
    const auto& ort_aliased_output_to_func_indices = func_info_->ort_aliased_output_to_func_indices;
    const auto& output_metas = func_info_->output_metas;

    for (const auto& p : ort_aliased_output_to_func_indices) {
      // p is a std::pair<int, size_t>. A pair of (ort dst idx, tvm src idx)
      // Purpose for using tvm src to avoid potential extra copying in kernel_compute_ctx
      int ort_output_idx = p.first;
      size_t tvm_idx = p.second;
      size_t tvm_output_idx = tvm_idx - func_info_->func_input_count;
      const TensorShape shape = TensorShape::FromExistingBuffer(dl_output_shapes[tvm_output_idx]);
      MLDataType dtype = output_metas[tvm_output_idx].dtype;
      void* dst = kernel_compute_ctx->OutputData(func_info_, ort_output_idx, shape, dtype);
      void* src = dl_tensors[tvm_idx].data;

      // TODO: change it to use provider::CopyTensor for non-CPU devices
      memcpy(dst, src, shape.Size() * dtype->Size());
    }
  }
}

void BasicExecBlock::InitContext(KernelComputeCtx* kernel_compute_ctx) const {
  const DLContext& dl_ctx = kernel_compute_ctx->GetRuntimeHandle()->dl_ctx;

  FuncComputeCtx& subgraph_compute_ctx = kernel_compute_ctx->GetFuncComputeCtx(func_info_);

  size_t tvm_input_count = func_info_->func_input_count;
  size_t tvm_output_count = func_info_->func_output_count;
  size_t tvm_num_args = tvm_input_count + tvm_output_count;

  std::vector<TVMValue>& lvalues = subgraph_compute_ctx.lvalues;
  lvalues.resize(tvm_num_args);
  std::vector<DLTensor>& dl_tensors = subgraph_compute_ctx.dl_tensors;
  dl_tensors.resize(tvm_num_args);
  std::vector<std::vector<int64_t>>& dl_output_shapes = subgraph_compute_ctx.dl_output_shapes;
  dl_output_shapes.resize(tvm_output_count);

  // a common lambda utility function for fill-in inputs
  auto fill_input = [&](size_t tvm_idx, const void* input_data, const int64_t* input_shape, size_t shape_rank, MLDataType data_type) {
    ORT_ENFORCE_DEBUG(kernel_compute_ctx->GetRuntimeHandle()->allow_unaligned_buffers ||
                      (reinterpret_cast<std::uintptr_t>(input_data)) % 64 == 0);
    DLDataType dtype = tvm_codegen::ToTvmDLDataType(data_type);
    dl_tensors[tvm_idx] = {const_cast<void*>(input_data), dl_ctx,
                           gsl::narrow_cast<int>(shape_rank), dtype,
                           const_cast<int64_t*>(input_shape), nullptr, 0};
    lvalues[tvm_idx].v_handle = &(dl_tensors[tvm_idx]);
  };

  // Handle Inputs (not including initializers)
  size_t tvm_input_idx = 0;
  for (const auto& input_meta : func_info_->input_metas) {
    int ort_input_idx = input_meta.ort_arg_index;
    const void* input_data = subgraph_compute_ctx.ort_input_data[ort_input_idx];
    const int64_t* input_shape = subgraph_compute_ctx.ort_input_shapes[ort_input_idx];
    MLDataType data_type = input_meta.dtype;

    // update dynamic shape in realized_dims
    const auto& symbols = input_meta.dim_symbols;
    kernel_compute_ctx->UpdateRealizedDims(symbols, input_shape);

    fill_input(tvm_input_idx++, input_data, input_shape, input_meta.inferred_shape.size(), data_type);
  }

  // Handle Initializers
  const std::vector<const Tensor*>& intializers = func_info_->intializers;
  for (const Tensor* t : intializers) {
    fill_input(tvm_input_idx++, t->DataRaw(), t->Shape().GetDims().data(),
               t->Shape().NumDimensions(), t->DataType());
  }

  // Handle Outputs
  size_t tvm_output_idx = 0;
  for (const auto& output_meta : func_info_->output_metas) {
    std::vector<int64_t>& realized_output_shape = dl_output_shapes[tvm_output_idx];
    // Update static dim
    realized_output_shape = output_meta.inferred_shape;
    // Update dynamic dim
    const std::vector<std::pair<size_t, std::string>>& symbols = output_meta.dim_symbols;
    kernel_compute_ctx->UpdateRealizedDims(symbols, realized_output_shape);

    int ort_output_idx = output_meta.ort_arg_index;

    // Fill in output DLTensor
    MLDataType data_type = output_meta.dtype;
    void* output_data = kernel_compute_ctx->OutputData(func_info_,
                                                       ort_output_idx,
                                                       TensorShape::FromExistingBuffer(realized_output_shape),
                                                       data_type);

    ORT_ENFORCE_DEBUG(kernel_compute_ctx->GetRuntimeHandle()->allow_unaligned_buffers ||
                      (reinterpret_cast<std::uintptr_t>(output_data)) % 64 == 0);

    DLDataType dtype = tvm_codegen::ToTvmDLDataType(data_type);

    size_t tvm_idx = tvm_output_idx + tvm_input_count;

    dl_tensors[tvm_idx] = {output_data, dl_ctx,
                           gsl::narrow_cast<int>(realized_output_shape.size()),
                           dtype, realized_output_shape.data(), nullptr, 0};
    lvalues[tvm_idx].v_handle = &(dl_tensors[tvm_idx]);

    ++tvm_output_idx;
  }
}

// UpdateContext is for an existing KernelComputeCtx, and only needs to update non-initializer input/output
void BasicExecBlock::UpdateContext(KernelComputeCtx* kernel_compute_ctx) const {
  FuncComputeCtx& subgraph_compute_ctx = kernel_compute_ctx->GetFuncComputeCtx(func_info_);
  std::vector<std::vector<int64_t>>& dl_output_shapes = subgraph_compute_ctx.dl_output_shapes;

  // Handle Inputs
  size_t tvm_input_idx = 0;
  for (const auto& input_meta : func_info_->input_metas) {
    int ort_input_idx = input_meta.ort_arg_index;

    // data ptr
    DLTensor& dl_tensor = subgraph_compute_ctx.dl_tensors[tvm_input_idx];
    dl_tensor.data = const_cast<void*>(subgraph_compute_ctx.ort_input_data[ort_input_idx]);
    const int64_t* input_shape = subgraph_compute_ctx.ort_input_shapes[ort_input_idx];
    // update dynamic shape in realized_dims
    const auto& symbols = input_meta.dim_symbols;
    kernel_compute_ctx->UpdateRealizedDims(symbols, input_shape);

    dl_tensor.shape = const_cast<int64_t*>(input_shape);
    dl_tensor.ndim = gsl::narrow<int>(input_meta.inferred_shape.size());
    ++tvm_input_idx;
  }

  // No need to update initializer in UpdateContext

  // Handle Outputs
  size_t tvm_output_idx = 0;
  for (const auto& output_meta : func_info_->output_metas) {
    size_t tvm_idx = tvm_output_idx + func_info_->func_input_count;
    DLTensor& dl_tensor = subgraph_compute_ctx.dl_tensors[tvm_idx];
    // Update dynamic dim
    const std::vector<std::pair<size_t, std::string>>& symbols = output_meta.dim_symbols;
    kernel_compute_ctx->UpdateRealizedDims(symbols, dl_output_shapes[tvm_output_idx]);

    int ort_output_idx = output_meta.ort_arg_index;

    // update pointer
    dl_tensor.data = kernel_compute_ctx->OutputData(func_info_,
                                                    ort_output_idx,
                                                    TensorShape::FromExistingBuffer(dl_output_shapes[tvm_output_idx]),
                                                    output_meta.dtype);
    ++tvm_output_idx;
  }
}

}  // namespace nuphar
}  // namespace onnxruntime
