// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/control_flow/scan_exec_ctx.h"

#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/utils.h"

#include "core/codegen/common/common.h"
#include "core/codegen/target/ort_tvm_utils.h"
#include "core/codegen/target/tvm_context.h"
#include "gsl/gsl_util"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

void ScanExecCtx::Advance(const ControlFlowInfo* cf_info) {
  ORT_ENFORCE_DEBUG(current_loop_step_ < max_loop_step_);
  // update inputs
  for (size_t scan_input_idx = 0; scan_input_idx < current_input_ptrs_.size(); ++scan_input_idx) {
    current_input_ptrs_[scan_input_idx] =
        (static_cast<char*>(current_input_ptrs_[scan_input_idx]) + input_strides_[scan_input_idx]);
  }

  // update outputs
  for (size_t scan_output_idx = 0; scan_output_idx < current_output_ptrs_.size(); ++scan_output_idx) {
    current_output_ptrs_[scan_output_idx] =
        (static_cast<char*>(current_output_ptrs_[scan_output_idx]) + output_strides_[scan_output_idx]);
  }

  const ScanExecInfo* scan_info = Promote<ScanExecInfo>(cf_info);
  ORT_ENFORCE_DEBUG(nullptr != scan_info);
  size_t num_state_variables = gsl::narrow<size_t>(scan_info->num_state_variables);
  const std::vector<int>& state_to_output_indices = scan_info->state_to_output_indices;

  // update input and output states
  if (current_loop_step_ == 0) {
    // When executed the first loop (current_loop_step == 0),
    // assign current_state_input_ptrs as current_state_output_ptrs
    // and current_state_output_ptrs as ort_state_output_buffers_
    for (size_t scan_state_idx = 0; scan_state_idx < num_state_variables; ++scan_state_idx) {
      current_ort_state_input_ptrs_[scan_state_idx] = current_ort_state_output_ptrs_[scan_state_idx];

      int out_idx = state_to_output_indices[scan_state_idx];
      if (out_idx >= 0) {
        current_ort_state_output_ptrs_[scan_state_idx] = current_output_ptrs_[out_idx];
      } else {
        current_ort_state_output_ptrs_[scan_state_idx] = ort_state_output_buffers_[scan_state_idx];
      }
    }
  } else if (current_loop_step_ == (max_loop_step_ - 1)) {
    // When executed the last loop step
    // copy from current_state_output_ptrs to state_output_ptrs if needed
    for (size_t scan_state_idx = 0; scan_state_idx < num_state_variables; ++scan_state_idx) {
      if (current_ort_state_output_ptrs_[scan_state_idx] != ort_state_output_buffers_[scan_state_idx])
        memcpy(ort_state_output_buffers_[scan_state_idx],
               current_ort_state_output_ptrs_[scan_state_idx],
               state_bytes_size_[scan_state_idx]);
    }
  } else {
    // When current_loop_step > 0
    // Swap current_ort_state_input_ptrs_[i] and current_ort_state_output_ptrs_[i]
    for (size_t scan_state_idx = 0; scan_state_idx < num_state_variables; ++scan_state_idx) {
      int scan_output_idx = state_to_output_indices[scan_state_idx];
      if (scan_output_idx >= 0) {
        current_ort_state_input_ptrs_[scan_state_idx] = current_ort_state_output_ptrs_[scan_state_idx];
        current_ort_state_output_ptrs_[scan_state_idx] = current_output_ptrs_[scan_output_idx];
      } else {
        std::swap(current_ort_state_input_ptrs_[scan_state_idx],
                  current_ort_state_output_ptrs_[scan_state_idx]);
      }
    }
  }

  // increase loop index
  ++current_loop_step_;
}

void ScanExecCtx::FillTVMArgs(NupharComputeCtx* compute_ctx) {
  std::vector<DLTensor>& dl_tensors = compute_ctx->GetDLTensors();
  const NupharFuncInfo& func_info = compute_ctx->GetNupharFuncInfo();

  size_t arg_index = 0;
  // update func inputs
  for (auto ptr : current_ort_state_input_ptrs_) {
    dl_tensors[arg_index].data = ptr;
    ++arg_index;
  }

  // update inputs
  for (auto ptr : current_input_ptrs_) {
    dl_tensors[arg_index].data = ptr;
    ++arg_index;
  }

  arg_index = func_info.input_count;
  // update func outputs
  for (auto ptr : current_func_output_ptrs_) {
    dl_tensors[arg_index].data = *ptr;
    ++arg_index;
  }
}

void ScanExecCtx::InitContext(NupharComputeCtx* compute_ctx) {
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

  // control flow info
  const ScanExecInfo* scan_info = Promote<ScanExecInfo>(func_info.cf_info.get());
  ORT_ENFORCE_DEBUG(nullptr != scan_info);
  int64_t num_state_variables = scan_info->num_state_variables;
  int64_t num_scan_inputs = scan_info->num_scan_inputs;
  int64_t num_scan_outputs = scan_info->num_scan_outputs;
  const std::vector<int64_t>& scan_input_axes = scan_info->scan_input_axes;
  const std::vector<int64_t>& scan_output_axes = scan_info->scan_output_axes;
  const std::vector<bool>& scan_input_forwards = scan_info->scan_input_forwards;
  const std::vector<bool>& scan_output_forwards = scan_info->scan_output_forwards;

  // a common lambda utility function for fill-in inputs and initializers
  auto fill_input = [&](size_t tvm_idx, const void* input_data, const int64_t* shape, size_t rank, MLDataType data_type) {
    ORT_ENFORCE_DEBUG(compute_ctx->GetRuntimeHandle()->allow_unaligned_buffers ||
                      (reinterpret_cast<std::uintptr_t>(input_data)) % 64 == 0);
    DLDataType dtype = tvm_codegen::ToTvmDLDataType(data_type);
    dl_tensors[tvm_idx] = {const_cast<void*>(input_data), dl_ctx,
                           gsl::narrow_cast<int>(rank), dtype,
                           const_cast<int64_t*>(shape), nullptr, 0};
    lvalues[tvm_idx].v_handle = &(dl_tensors[tvm_idx]);
  };

  // Reserve sizes of Scan's fast look ptr
  current_input_ptrs_.resize(num_scan_inputs);
  input_strides_.resize(num_scan_inputs);
  current_output_ptrs_.resize(num_scan_outputs);
  output_strides_.resize(num_scan_outputs);
  current_ort_state_input_ptrs_.resize(num_state_variables);
  current_ort_state_output_ptrs_.resize(num_state_variables);
  ort_state_input_buffers_.resize(num_state_variables);
  ort_state_output_buffers_.resize(num_state_variables);

  scan_input_in_subgraph_shapes_.resize(num_scan_inputs);
  scan_output_shapes_.resize(num_scan_outputs);

  state_bytes_size_.resize(num_state_variables);

  // Handle Scan's control flow ctx
  seq_length_ = 0;
  for (int ort_input_idx = gsl::narrow<int>(num_state_variables);
       ort_input_idx < compute_ctx->InputCount();
       ++ort_input_idx) {
    const int64_t* input_shape = compute_ctx->InputShape(ort_input_idx);
    size_t scan_input_idx = gsl::narrow<size_t>(ort_input_idx) - gsl::narrow<size_t>(num_state_variables);
    ORT_ENFORCE_DEBUG(scan_input_idx >= 0 && scan_input_idx < scan_input_axes.size());
    size_t input_scan_axis = gsl::narrow<size_t>(scan_input_axes[scan_input_idx]);
    if (seq_length_ == 0)
      seq_length_ = input_shape[input_scan_axis];
    ORT_ENFORCE_DEBUG(seq_length_ == input_shape[input_scan_axis]);
  }

  min_loop_step_ = 0;
  max_loop_step_ = seq_length_;
  current_loop_step_ = min_loop_step_;

  // Handle Inputs (not including initializers)
  // Input meta
  const auto& input_metas = func_info.input_metas;
  const std::vector<int>& ort_input_to_func_indices = func_info.ort_input_to_func_indices;
  // Assign inputs
  size_t mutable_input_count = 0;  // inputs that are not initializers
  for (int ort_input_idx = 0; ort_input_idx < compute_ctx->InputCount(); ++ort_input_idx) {
    int tvm_idx = ort_input_to_func_indices[ort_input_idx];

    // Skip initializer
    if (tvm_idx < 0)
      continue;

    mutable_input_count++;

    size_t tvm_input_idx = gsl::narrow<size_t>(tvm_idx);
    const void* input_data = compute_ctx->InputData(ort_input_idx);
    const int64_t* ort_input_shape = compute_ctx->InputShape(ort_input_idx);
    const auto& input_meta = input_metas[tvm_input_idx];
    MLDataType data_type = input_meta.dtype;
    size_t arg_shape_rank = input_meta.inferred_shape.size();

    if (ort_input_idx < gsl::narrow<int>(num_state_variables)) {
      ORT_ENFORCE_DEBUG(compute_ctx->InputShapeRank(ort_input_idx) == arg_shape_rank);
      // for scan_state_input, the rank is the same
      fill_input(tvm_input_idx, input_data, ort_input_shape, arg_shape_rank, data_type);
      // set the input_data, which is the main graph's state input ptr, to current current_ort_state_input_ptrs_
      current_ort_state_input_ptrs_[ort_input_idx] = dl_tensors[tvm_input_idx].data;
      // set the new allocated ptr to state_input_buffers
      ort_state_buffer_unique_ptrs_.push_back(compute_ctx->AllocateData(ort_input_shape, arg_shape_rank, data_type));
      ort_state_input_buffers_[ort_input_idx] = ort_state_buffer_unique_ptrs_.back().get();
    } else {
      ORT_ENFORCE_DEBUG(compute_ctx->InputShapeRank(ort_input_idx) == arg_shape_rank + 1);
      // if ith varialbe is an input, we need to slice it based on the scan_input_axes
      size_t scan_input_idx = gsl::narrow<size_t>(ort_input_idx) - gsl::narrow<size_t>(num_state_variables);
      ORT_ENFORCE_DEBUG(scan_input_idx >= 0 && scan_input_idx < scan_input_axes.size());
      size_t input_scan_axis = gsl::narrow<size_t>(scan_input_axes[scan_input_idx]);

      std::vector<int64_t>& shape = scan_input_in_subgraph_shapes_[scan_input_idx];
      ShapeRemoveAxis(shape, ort_input_shape, arg_shape_rank + 1, input_scan_axis);

      // Check whether it is backward Scan
      // If so, we need to use the last frame, instead of the first frame.
      int64_t stride = BytesOfShape(shape, data_type);
      input_data = scan_input_forwards[scan_input_idx]
                       ? input_data
                       : (static_cast<const char*>(input_data) + stride * (seq_length_ - 1));

      fill_input(tvm_input_idx, input_data, shape.data(), shape.size(), data_type);

      // set the input_data, which is the main graph's input ptr
      current_input_ptrs_[scan_input_idx] = dl_tensors[tvm_input_idx].data;
      // use sliced shape and data_type as stride
      input_strides_[scan_input_idx] = scan_input_forwards[scan_input_idx] ? stride : -stride;
    }

    // update dynamic shape in realized_dims
    // TODO: move this piece of code to compute_ctx
    const auto& symbols = input_meta.dim_symbols;
    for (const auto& s_pair : symbols) {
      size_t tvm_dim = s_pair.first;
      int64_t dim_size = dl_tensors[tvm_input_idx].shape[tvm_dim];
      const std::string& dim_param = s_pair.second;
      auto dim_value_iter = realized_dims.find(dim_param);

      if (dim_value_iter == realized_dims.end()) {
        realized_dims.insert(std::make_pair(dim_param, dim_size));  // update new symbol
      } else if (dim_value_iter->second == Dimension_Unknown) {
        dim_value_iter->second = dim_size;  // update for a symbol
      } else {
        // a true runtime error, so use ORT_ENFORCE, not ORT_ENFORCE_DEBUG
        ORT_ENFORCE(dim_value_iter->second == dim_size);
      }
    }
  }

  // Handle Initializers
  // Initializer meta
  const std::vector<const Tensor*>& intializers = func_info.intializers;

  // Assign Initializers
  size_t tvm_input_idx = mutable_input_count;
  for (const Tensor* t : intializers) {
    fill_input(tvm_input_idx++, t->DataRaw(), t->Shape().GetDims().data(), t->Shape().NumDimensions(), t->DataType());
  }

  // Handle outputs and state outputs
  // Output meta
  const auto& output_metas = func_info.output_metas;
  const std::vector<int>& ort_output_to_func_indices = func_info.ort_output_to_func_indices;
  const std::vector<std::pair<int, size_t>>& ort_aliased_output_to_func_indices = func_info.ort_aliased_output_to_func_indices;

  current_func_output_ptrs_.resize(tvm_output_count);

  // Assign outputs and state outputs
  for (int ort_output_idx = 0; ort_output_idx < compute_ctx->OutputCount(); ++ort_output_idx) {
    int tvm_idx = ort_output_to_func_indices[ort_output_idx];

    // skip aliased output
    if (tvm_idx < 0)
      continue;

    size_t tvm_output_idx = tvm_idx - tvm_input_count;
    const auto& output_meta = output_metas[tvm_output_idx];
    // TODO: move this to compute_ctx
    std::vector<int64_t>& realized_shape = dl_output_shapes[tvm_output_idx];

    // Update static dim
    realized_shape = output_meta.inferred_shape;

    // Update dynamic dim
    // TODO: move this piece of code to compute_ctx
    const std::vector<std::pair<size_t, std::string>>& symbols = output_meta.dim_symbols;
    for (const auto& s_pair : symbols) {
      size_t dim = s_pair.first;
      const std::string& dim_param = s_pair.second;
      auto dim_value_iter = realized_dims.find(dim_param);
      ORT_ENFORCE_DEBUG(dim_value_iter != realized_dims.end());
      realized_shape[dim] = dim_value_iter->second;
    }

    // Fill in output DLTensor
    MLDataType data_type = output_meta.dtype;  // static meta from NupharFuncInfo
    void* output_data = nullptr;
    // if i variables is smaller than num_state_variables, check whether it is an output or a state output
    if (ort_output_idx < gsl::narrow<int>(num_state_variables)) {
      //  if ith variable is a state output, we just call OutputData2 API with realized_shape
      output_data = compute_ctx->OutputData(ort_output_idx, TensorShape::ReinterpretBaseType(realized_shape));

      // set current_ort_state_output_ptrs_ as ort_state_input_buffers_
      // Note it is "ort_state_input_buffers_", since we will perform double buffering later.
      current_ort_state_output_ptrs_[ort_output_idx] = ort_state_input_buffers_[ort_output_idx];
      // set ort_state_output_buffers_ as output_data
      ort_state_output_buffers_[ort_output_idx] = output_data;
      state_bytes_size_[ort_output_idx] = BytesOfShape(realized_shape, data_type);

      // link current_func_output_ptrs_ as current_ort_state_output_ptrs_
      current_func_output_ptrs_[ort_output_idx] = &current_ort_state_output_ptrs_[ort_output_idx];
    } else {
      // if ith varialbe is an output, we need to remove an axis for DLTesnor
      size_t scan_output_idx = gsl::narrow<size_t>(ort_output_idx) - gsl::narrow<size_t>(num_state_variables);
      std::vector<int64_t>& shape = scan_output_shapes_[scan_output_idx];
      ORT_ENFORCE_DEBUG(scan_output_idx >= 0 && scan_output_idx < scan_output_axes.size());
      size_t output_scan_axis = gsl::narrow<size_t>(scan_output_axes[scan_output_idx]);
      ShapeInsertAxis(shape, realized_shape.data(), realized_shape.size(), output_scan_axis, seq_length_);

      output_data = compute_ctx->OutputData(ort_output_idx, TensorShape::ReinterpretBaseType(shape));

      // Check whether it is backward Scan
      // If so, we need to use the last frame, instead of the first frame.
      // Note here sliced_shape is realized_shape, since realized_shape is from NupharFunctionInfo.
      int64_t stride = BytesOfShape(realized_shape, data_type);
      output_data = scan_output_forwards[scan_output_idx]
                        ? output_data
                        : (static_cast<char*>(output_data) + stride * (seq_length_ - 1));

      // set output_data to current_output_ptrs_
      current_output_ptrs_[scan_output_idx] = output_data;
      // use sliced shape and data_type as stride
      output_strides_[scan_output_idx] = scan_output_forwards[scan_output_idx] ? stride : -stride;

      // link current_func_output_ptrs_ to current_output_ptrs_
      size_t tvm_output_idx = gsl::narrow<size_t>(tvm_idx) - tvm_input_count;
      current_func_output_ptrs_[tvm_output_idx] = &current_output_ptrs_[scan_output_idx];
    }

    ORT_ENFORCE_DEBUG(compute_ctx->GetRuntimeHandle()->allow_unaligned_buffers ||
                      (reinterpret_cast<std::uintptr_t>(output_data)) % 64 == 0);
    DLDataType dtype = tvm_codegen::ToTvmDLDataType(data_type);
    dl_tensors[tvm_idx] = {output_data, dl_ctx, gsl::narrow<int>(realized_shape.size()),
                           dtype, realized_shape.data(), nullptr, 0};
    lvalues[tvm_idx].v_handle = &(dl_tensors[tvm_idx]);
  }

  // Handle alias state outputs
  for (const auto& p : ort_aliased_output_to_func_indices) {
    // p is a std::pair<int, size_t>. A pair of (ort dst idx, tvm src idx)
    // Note ort dst idx is always a state output
    int ort_state_idx = p.first;
    size_t tvm_idx = p.second;
    size_t tvm_output_idx = tvm_idx - tvm_input_count;
    MLDataType data_type = output_metas[tvm_output_idx].dtype;
    current_ort_state_output_ptrs_[ort_state_idx] = dl_tensors[tvm_idx].data;
    ort_state_output_buffers_[ort_state_idx] =
        compute_ctx->OutputData(ort_state_idx, TensorShape::ReinterpretBaseType(dl_output_shapes[tvm_output_idx]));
    state_bytes_size_[ort_state_idx] = BytesOfShape(dl_output_shapes[tvm_output_idx], data_type);
  }
}

// UpdateContext is for an existing NupharComputeCtx, and only needs to update non-initializer input/output
void ScanExecCtx::UpdateContext(NupharComputeCtx* compute_ctx) {
  // control flow info
  const NupharFuncInfo& func_info = compute_ctx->GetNupharFuncInfo();
  std::unordered_map<std::string, int64_t>& realized_dims = compute_ctx->GetRealizedDims();

  size_t tvm_input_count = func_info.input_count;

  // control flow info
  const ScanExecInfo* scan_info = Promote<ScanExecInfo>(func_info.cf_info.get());
  ORT_ENFORCE_DEBUG(nullptr != scan_info);
  int64_t num_state_variables = scan_info->num_state_variables;
  const std::vector<int64_t>& scan_input_axes = scan_info->scan_input_axes;
  const std::vector<bool>& scan_input_forwards = scan_info->scan_input_forwards;
  const std::vector<bool>& scan_output_forwards = scan_info->scan_output_forwards;
  const std::vector<int64_t>& scan_output_axes = scan_info->scan_output_axes;

  // Handle Scan's control flow ctx
  seq_length_ = 0;
  for (int ort_input_idx = gsl::narrow<int>(num_state_variables);
       ort_input_idx < compute_ctx->InputCount();
       ++ort_input_idx) {
    const int64_t* ort_input_shape = compute_ctx->InputShape(ort_input_idx);
    size_t scan_input_idx = gsl::narrow<size_t>(ort_input_idx) - gsl::narrow<size_t>(num_state_variables);
    ORT_ENFORCE_DEBUG(scan_input_idx >= 0 && scan_input_idx < scan_input_axes.size());
    size_t input_scan_axis = gsl::narrow<size_t>(scan_input_axes[scan_input_idx]);
    if (seq_length_ == 0)
      seq_length_ = ort_input_shape[input_scan_axis];
    ORT_ENFORCE_DEBUG(seq_length_ == ort_input_shape[input_scan_axis]);
  }
  min_loop_step_ = 0;
  max_loop_step_ = seq_length_;
  current_loop_step_ = min_loop_step_;

  // Handle inputs and state inputs (not including initializer)
  // Input meta
  const auto& input_metas = func_info.input_metas;
  const std::vector<int>& ort_input_to_func_indices = func_info.ort_input_to_func_indices;

  // Assign inputs and state
  for (int ort_input_idx = 0; ort_input_idx < compute_ctx->InputCount(); ++ort_input_idx) {
    int tvm_idx = ort_input_to_func_indices[ort_input_idx];

    // Skip initializer
    if (tvm_idx < 0)
      continue;

    size_t tvm_input_idx = gsl::narrow<size_t>(tvm_idx);
    DLTensor& dl_tensor = compute_ctx->GetDLTensor(tvm_idx);
    const auto& input_meta = input_metas[tvm_input_idx];

    const int64_t* ort_input_shape = compute_ctx->InputShape(ort_input_idx);
    const auto& symbols = input_meta.dim_symbols;
    MLDataType data_type = input_meta.dtype;

    // check whether it is an input or a state input
    int scan_input_idx = ort_input_idx - gsl::narrow<int>(num_state_variables);
    size_t input_scan_axis = 0;
    bool is_scan_input = (scan_input_idx >= 0);
    if (is_scan_input) {
      ORT_ENFORCE_DEBUG(scan_input_idx < scan_input_axes.size());
      input_scan_axis = gsl::narrow<size_t>(scan_input_axes[scan_input_idx]);
    }

    // update scan inputs' dynamic shape in realized_dims
    // state input would use shape from ort directly
    // TODO: move this piece of code to compute_ctx
    if (is_scan_input) {
      for (const auto& s_pair : symbols) {
        size_t tvm_dim = s_pair.first;
        size_t ort_dim = tvm_dim;
        if (tvm_dim >= input_scan_axis) {
          ort_dim = tvm_dim + 1;
        }
        int64_t dim_size = ort_input_shape[ort_dim];
        auto dim_value_iter = realized_dims.find(s_pair.second);
        ORT_ENFORCE_DEBUG(dim_value_iter != realized_dims.end());

        if (dim_value_iter->second == Dimension_Unknown) {
          dim_value_iter->second = dim_size;
        } else {
          // a true runtime error, so use ORT_ENFORCE, not ORT_ENFORCE_DEBUG
          ORT_ENFORCE(dim_value_iter->second == dim_size);
        }

        dl_tensor.shape[tvm_dim] = dim_size;
      }
    }

    // update ptr

    void* input_data = const_cast<void*>(compute_ctx->InputData(ort_input_idx));

    if (is_scan_input) {
      ORT_ENFORCE_DEBUG(compute_ctx->InputShapeRank(ort_input_idx) == input_meta.inferred_shape.size() + 1);
      // if ith varialbe is an input, we need to use sliced shape (from dl_tensor.shape)
      int64_t stride = BytesOfShape(dl_tensor.shape, dl_tensor.ndim, data_type);
      // Check whether it is backward Scan
      // If so, we need to use the last frame, instead of the first frame.
      input_data = scan_input_forwards[scan_input_idx]
                       ? input_data
                       : (static_cast<char*>(input_data) + stride * (seq_length_ - 1));

      size_t scan_input_idx = tvm_input_idx - gsl::narrow<size_t>(num_state_variables);
      dl_tensor.data = input_data;
      current_input_ptrs_[scan_input_idx] = input_data;
      // use sliced shape and data_type as stride
      input_strides_[scan_input_idx] = scan_input_forwards[scan_input_idx] ? stride : -stride;
    } else {
      ORT_ENFORCE_DEBUG(compute_ctx->InputShapeRank(ort_input_idx) == input_meta.inferred_shape.size());
      // if ith variable is a state input
      // set the input_data, which is the main graph's state input ptr, to current current_ort_state_input_ptrs_
      dl_tensor.data = input_data;
      dl_tensor.shape = const_cast<int64_t*>(ort_input_shape);
      dl_tensor.ndim = gsl::narrow<int>(input_meta.inferred_shape.size());
      current_ort_state_input_ptrs_[ort_input_idx] = input_data;
      // set the new allocated ptr to state_input_buffers
      ort_state_buffer_unique_ptrs_.push_back(compute_ctx->AllocateData(dl_tensor.shape, dl_tensor.ndim, data_type));
      ort_state_input_buffers_[ort_input_idx] = ort_state_buffer_unique_ptrs_.back().get();
    }
  }

  // Handle outputs and state outputs
  // Output meta
  const auto& output_metas = func_info.output_metas;
  const std::vector<int>& ort_output_to_func_indices = func_info.ort_output_to_func_indices;

  // Assign Outputs
  std::vector<std::vector<int64_t>>& dl_output_shapes = compute_ctx->GetDLOutputShapes();
  for (int ort_output_idx = 0; ort_output_idx < compute_ctx->OutputCount(); ++ort_output_idx) {
    int tvm_idx = ort_output_to_func_indices[ort_output_idx];

    // Skip aliased output
    if (tvm_idx < 0)
      continue;

    size_t tvm_output_idx = tvm_idx - tvm_input_count;
    const auto& output_meta = output_metas[tvm_output_idx];
    MLDataType data_type = output_meta.dtype;
    DLTensor& dl_tensor = compute_ctx->GetDLTensor(tvm_idx);

    int scan_output_idx = ort_output_idx - gsl::narrow<int>(num_state_variables);
    size_t output_scan_axis = 0;
    bool is_scan_output = (scan_output_idx >= 0);
    if (is_scan_output) {
      ORT_ENFORCE_DEBUG(scan_output_idx < scan_output_axes.size());
      output_scan_axis = gsl::narrow<size_t>(scan_output_axes[scan_output_idx]);
    }
    std::vector<int64_t>& ort_output_shape =
        is_scan_output ? scan_output_shapes_[scan_output_idx] : dl_output_shapes[tvm_output_idx];

    // Update dynamic dim
    // TODO: move this piece of code to compute_ctx
    const auto& symbols = output_meta.dim_symbols;
    for (const auto& s_pair : symbols) {
      size_t tvm_dim = s_pair.first;
      size_t ort_dim = tvm_dim;
      if (is_scan_output && tvm_dim > output_scan_axis) {
        ort_dim = tvm_dim + 1;
      }
      const std::string& dim_param = s_pair.second;
      auto dim_value_iter = realized_dims.find(dim_param);
      ORT_ENFORCE_DEBUG(dim_value_iter != realized_dims.end());
      // update output shapes for tvm and ort
      ort_output_shape[ort_dim] = dim_value_iter->second;
      dl_tensor.shape[tvm_dim] = dim_value_iter->second;
    }

    // update ptr
    void* output_data = nullptr;
    if (ort_output_idx < gsl::narrow<int>(num_state_variables)) {
      output_data = compute_ctx->OutputData(ort_output_idx, TensorShape::ReinterpretBaseType(ort_output_shape));
      // set current_ort_state_output_ptrs_ as ort_state_input_buffers_
      // Note it is "ort_state_input_buffers_", since we will perform double buffering later.
      current_ort_state_output_ptrs_[ort_output_idx] = ort_state_input_buffers_[ort_output_idx];
      // set ort_state_output_buffers_ as output_data
      ort_state_output_buffers_[ort_output_idx] = output_data;
      state_bytes_size_[ort_output_idx] = BytesOfShape(ort_output_shape, data_type);
    } else {
      ort_output_shape[output_scan_axis] = seq_length_;
      output_data = compute_ctx->OutputData(ort_output_idx, TensorShape::ReinterpretBaseType(ort_output_shape));
      // Check whether it is backward Scan
      // If so, we need to use the last frame, instead of the first frame.
      // Note here stride come from dl_tensor shape
      int64_t stride = BytesOfShape(dl_tensor.shape, dl_tensor.ndim, data_type);
      output_data = scan_output_forwards[scan_output_idx]
                        ? output_data
                        : (static_cast<char*>(output_data) + stride * (seq_length_ - 1));

      // set output_data to current_output_ptrs_
      current_output_ptrs_[scan_output_idx] = output_data;
      // use sliced shape and data_type as stride
      output_strides_[scan_output_idx] = scan_output_forwards[scan_output_idx] ? stride : -stride;
    }

    dl_tensor.data = output_data;
  }

  // Handle alias state outputs
  const std::vector<std::pair<int, size_t>>& ort_aliased_output_to_func_indices = func_info.ort_aliased_output_to_func_indices;
  for (const auto& p : ort_aliased_output_to_func_indices) {
    // p is a std::pair<int, size_t>. A pair of (ort dst idx, tvm src idx)
    // Note ort dst idx is always a state output
    int ort_state_idx = p.first;
    size_t tvm_idx = p.second;
    size_t tvm_output_idx = tvm_idx - tvm_input_count;
    MLDataType data_type = output_metas[tvm_output_idx].dtype;
    DLTensor& dl_tensor = compute_ctx->GetDLTensor(tvm_idx);
    current_ort_state_output_ptrs_[ort_state_idx] = dl_tensor.data;
    ort_state_output_buffers_[ort_state_idx] =
        compute_ctx->OutputData(ort_state_idx, TensorShape::ReinterpretBaseType(dl_output_shapes[tvm_output_idx]));
    state_bytes_size_[ort_state_idx] = BytesOfShape(dl_output_shapes[tvm_output_idx], data_type);
  }
}

void ScanExecCtx::LoopFinalize() {
  seq_length_ = 0;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
