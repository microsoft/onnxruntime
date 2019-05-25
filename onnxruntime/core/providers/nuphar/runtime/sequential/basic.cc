// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/sequential/basic.h"

#include "core/codegen/common/common.h"
#include "core/codegen/target/ort_tvm_utils.h"
#include "gsl/gsl_util"
#include <tvm/tvm.h>

// from onnxruntime_typeinf.cc, in global namespace
const onnxruntime::DataTypeImpl* ElementTypeFromProto(int type);

namespace onnxruntime {
namespace tvm_codegen {

void BasicExecBlock::Run(NupharComputeCtx* compute_ctx) {
  if (compute_ctx->HasInitialized()) {
    UpdateContext(compute_ctx);
  } else {
    InitContext(compute_ctx);
  }

  const NupharFuncInfo& func_info = compute_ctx->GetNupharFuncInfo();

  size_t tvm_input_count = func_info.input_count;
  size_t tvm_output_count = func_info.output_count;
  int tvm_num_args = gsl::narrow<int>(tvm_input_count + tvm_output_count);
  tvm::TVMArgs tvm_args(compute_ctx->GetTVMValues().data(),
                        func_info.type_codes.data(),
                        tvm_num_args);

  tvm::TVMRetValue rvalue;
  const tvm::runtime::PackedFunc& func = func_info.packed_func;
  func.CallPacked(tvm_args, &rvalue);

  // Check aliased outputs
  if (tvm_output_count < compute_ctx->OutputCount()) {
    const std::vector<DLTensor>& dl_tensors = compute_ctx->GetDLTensors();
    const std::vector<std::vector<int64_t>>& dl_output_shapes = compute_ctx->GetDLOutputShapes();
    const auto& ort_aliased_output_to_func_indices = func_info.ort_aliased_output_to_func_indices;
    const auto& output_metas = func_info.output_metas;

    for (const auto& p : ort_aliased_output_to_func_indices) {
      // p is a std::pair<int, size_t>. A pair of (ort dst idx, tvm src idx)
      // Purpose for using tvm src to avoid potential extra copying in compute_ctx
      int ort_output_idx = p.first;
      size_t tvm_idx = p.second;
      size_t tvm_output_idx = tvm_idx - func_info.input_count;
      const TensorShape& shape = TensorShape::ReinterpretBaseType(dl_output_shapes[tvm_output_idx]);
      MLDataType dtype = output_metas[tvm_output_idx].dtype;
      void* dst = compute_ctx->OutputData(ort_output_idx, shape);
      void* src = dl_tensors[tvm_idx].data;

      // TODO: change it to use provider::CopyTensor for non-CPU devices
      memcpy(dst, src, shape.Size() * dtype->Size());
    }
  }
}

void BasicExecBlock::InitContext(NupharComputeCtx* compute_ctx) {
  const DLContext& dl_ctx = compute_ctx->GetRuntimeHandle()->dl_ctx;
  const NupharFuncInfo& func_info = compute_ctx->GetNupharFuncInfo();

  size_t tvm_input_count = func_info.input_count;
  size_t tvm_output_count = func_info.output_count;
  size_t tvm_num_args = tvm_input_count + tvm_output_count;

  std::vector<TVMValue>& lvalues = compute_ctx->GetTVMValues();
  lvalues.resize(tvm_num_args);
  std::vector<DLTensor>& dl_tensors = compute_ctx->GetDLTensors();
  dl_tensors.resize(tvm_num_args);
  std::vector<std::vector<int64_t>>& dl_output_shapes = compute_ctx->GetDLOutputShapes();
  dl_output_shapes.resize(tvm_output_count);

  std::unordered_map<std::string, int64_t>& realized_dims = compute_ctx->GetRealizedDims();

  // a common lambda utility function for fill-in inputs
  auto fill_input = [&](size_t tvm_idx, const void* input_data, const int64_t* input_shape, size_t shape_rank, MLDataType data_type) {
    ORT_ENFORCE_DEBUG(compute_ctx->GetRuntimeHandle()->allow_unaligned_buffers ||
                      (reinterpret_cast<std::uintptr_t>(input_data)) % 64 == 0);
    DLDataType dtype = tvm_codegen::ToTvmDLDataType(data_type);
    dl_tensors[tvm_idx] = {const_cast<void*>(input_data), dl_ctx,
                           gsl::narrow_cast<int>(shape_rank), dtype,
                           const_cast<int64_t*>(input_shape), nullptr, 0};
    lvalues[tvm_idx].v_handle = &(dl_tensors[tvm_idx]);
  };

  // Handle Inputs (not including initializers)
  // Input meta
  const auto& input_metas = func_info.input_metas;
  const std::vector<int>& ort_input_to_func_indices = func_info.ort_input_to_func_indices;

  // Assign inputs
  size_t mutable_input_count = 0;
  for (int ort_input_idx = 0; ort_input_idx < compute_ctx->InputCount(); ++ort_input_idx) {
    int tvm_idx = ort_input_to_func_indices[ort_input_idx];

    // Skip initializer
    if (tvm_idx < 0)
      continue;

    mutable_input_count++;

    size_t tvm_input_idx = gsl::narrow<size_t>(tvm_idx);
    const auto& input_meta = input_metas[tvm_input_idx];
    const void* input_data = compute_ctx->InputData(ort_input_idx);
    const int64_t* input_shape = compute_ctx->InputShape(ort_input_idx);
    MLDataType data_type = input_meta.dtype;
    fill_input(tvm_input_idx, input_data, input_shape, input_meta.inferred_shape.size(), data_type);

    // update dynamic shape in realized_dims
    // TODO: move this piece of code to compute_ctx
    const auto& symbols = input_meta.dim_symbols;
    for (const auto& s_pair : symbols) {
      size_t dim = s_pair.first;
      const std::string& dim_param = s_pair.second;
      auto dim_value_iter = realized_dims.find(dim_param);

      if (dim_value_iter == realized_dims.end()) {
        realized_dims.insert(std::make_pair(dim_param, input_shape[dim]));  // update new symbol
      } else if (dim_value_iter->second == Dimension_Unknown) {
        dim_value_iter->second = input_shape[dim];  // update for a symbol
      } else {
        ORT_ENFORCE(dim_value_iter->second == input_shape[dim]);  // a runtime error
      }
    }
  }

  // Handle Initializers
  // Initializer meta
  const std::vector<const Tensor*>& intializers = func_info.intializers;

  // Assign Initializers
  size_t tvm_input_idx = mutable_input_count;
  for (const Tensor* t : intializers) {
    fill_input(tvm_input_idx++, t->DataRaw(), t->Shape().GetDims().data(),
               t->Shape().NumDimensions(), t->DataType());
  }

  // Handle Outputs
  // Output meta
  const auto& output_metas = func_info.output_metas;
  const std::vector<int>& ort_output_to_func_indices = func_info.ort_output_to_func_indices;

  // Assign outputs
  for (int ort_output_idx = 0; ort_output_idx < compute_ctx->OutputCount(); ++ort_output_idx) {
    int tvm_idx = ort_output_to_func_indices[ort_output_idx];

    // skip aliased output
    if (tvm_idx < 0)
      continue;

    size_t tvm_output_idx = tvm_idx - tvm_input_count;
    const auto& output_meta = output_metas[tvm_output_idx];

    // TODO: move this to compute_ctx
    std::vector<int64_t>& realized_output_shape = dl_output_shapes[tvm_output_idx];
    // Update static dim
    realized_output_shape = output_meta.inferred_shape;

    // Update dynamic dim
    // TODO: move this piece of code to compute_ctx
    const std::vector<std::pair<size_t, std::string>>& symbols = output_meta.dim_symbols;
    for (const auto& s_pair : symbols) {
      size_t dim = s_pair.first;
      const std::string& dim_param = s_pair.second;
      auto dim_value_iter = realized_dims.find(dim_param);
      ORT_ENFORCE_DEBUG(dim_value_iter != realized_dims.end());
      realized_output_shape[dim] = dim_value_iter->second;
    }

    // Fill in output DLTensor
    MLDataType data_type = output_meta.dtype;
    void* output_data = compute_ctx->OutputData(ort_output_idx, TensorShape::ReinterpretBaseType(realized_output_shape));
    ORT_ENFORCE_DEBUG(compute_ctx->GetRuntimeHandle()->allow_unaligned_buffers || (reinterpret_cast<std::uintptr_t>(output_data)) % 64 == 0);
    DLDataType dtype = tvm_codegen::ToTvmDLDataType(data_type);
    dl_tensors[tvm_idx] = {output_data, dl_ctx,
                           gsl::narrow_cast<int>(realized_output_shape.size()),
                           dtype, realized_output_shape.data(), nullptr, 0};
    lvalues[tvm_idx].v_handle = &(dl_tensors[tvm_idx]);
  }
}

// UpdateContext is for an existing NupharComputeCtx, and only needs to update non-initializer input/output
void BasicExecBlock::UpdateContext(NupharComputeCtx* compute_ctx) {
  const NupharFuncInfo& func_info = compute_ctx->GetNupharFuncInfo();
  std::unordered_map<std::string, int64_t>& realized_dims = compute_ctx->GetRealizedDims();
  std::vector<std::vector<int64_t>>& dl_output_shapes = compute_ctx->GetDLOutputShapes();

  // Handle Inputs
  // Input meta
  const std::vector<int>& ort_input_to_func_indices = func_info.ort_input_to_func_indices;
  const auto& input_metas = func_info.input_metas;

  // Assign Inputs
  for (int ort_input_idx = 0; ort_input_idx < compute_ctx->InputCount(); ++ort_input_idx) {
    int tvm_idx = ort_input_to_func_indices[ort_input_idx];

    // Skip initializer
    if (tvm_idx < 0)
      continue;

    // data ptr
    DLTensor& dl_tensor = compute_ctx->GetDLTensor(tvm_idx);
    dl_tensor.data = const_cast<void*>(compute_ctx->InputData(ort_input_idx));

    size_t tvm_input_idx = gsl::narrow<size_t>(tvm_idx);
    const auto& input_meta = input_metas[tvm_input_idx];
    const int64_t* input_shape = compute_ctx->InputShape(ort_input_idx);
    dl_tensor.shape = const_cast<int64_t*>(input_shape);
    dl_tensor.ndim = gsl::narrow<int>(input_meta.inferred_shape.size());

    // update dynamic shape in realized_dims
    // TODO: move this piece of code to compute_ctx
    const auto& symbols = input_meta.dim_symbols;
    for (const auto& s_pair : symbols) {
      size_t dim = s_pair.first;
      int64_t dim_size = input_shape[dim];
      auto dim_value_iter = realized_dims.find(s_pair.second);
      ORT_ENFORCE_DEBUG(dim_value_iter != realized_dims.end());

      if (dim_value_iter->second == Dimension_Unknown) {
        // not set in current execution_frame, so update
        dim_value_iter->second = dim_size;
      } else {
        // a true runtime error, so use ORT_ENFORCE, not ORT_ENFORCE_DEBUG
        ORT_ENFORCE(dim_value_iter->second == dim_size);
      }
    }
  }

  // Handle Outputs
  // Output meta
  const auto& output_metas = func_info.output_metas;
  const std::vector<int>& ort_output_to_func_indices = func_info.ort_output_to_func_indices;

  // Assign Outputs
  for (int ort_output_idx = 0; ort_output_idx < compute_ctx->OutputCount(); ++ort_output_idx) {
    int tvm_idx = ort_output_to_func_indices[ort_output_idx];

    // Skip aliased output
    if (tvm_idx < 0)
      continue;

    size_t tvm_output_idx = tvm_idx - func_info.input_count;
    DLTensor& dl_tensor = compute_ctx->GetDLTensor(tvm_idx);
    const auto& output_meta = output_metas[tvm_output_idx];

    // Update dynamic dim
    // TODO: move this piece of code to compute_ctx
    const std::vector<std::pair<size_t, std::string>>& symbols = output_meta.dim_symbols;
    for (const auto& s_pair : symbols) {
      auto dim_value_iter = realized_dims.find(s_pair.second);
      ORT_ENFORCE_DEBUG(dim_value_iter != realized_dims.end());
      dl_output_shapes[tvm_output_idx][s_pair.first] = dim_value_iter->second;
    }

    // update pointer
    dl_tensor.data = compute_ctx->OutputData(
        ort_output_idx,
        TensorShape::ReinterpretBaseType(dl_output_shapes[tvm_output_idx]));
  }
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
