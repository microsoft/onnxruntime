// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/tensorscatter.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/tensor.h"

#include <cstring>

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    TensorScatter,
    24,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    TensorScatter);

TensorScatter::TensorScatter(const OpKernelInfo& info) : OpKernel(info) {
  axis_ = info.GetAttrOrDefault<int64_t>("axis", -2);
  std::string mode = info.GetAttrOrDefault<std::string>("mode", "linear");
  ORT_ENFORCE(mode == "linear" || mode == "circular",
              "TensorScatter: mode must be 'linear' or 'circular', got '", mode, "'");
  circular_ = (mode == "circular");
}

Status TensorScatter::Compute(OpKernelContext* context) const {
  const Tensor* past_cache = context->Input<Tensor>(0);
  const Tensor* update = context->Input<Tensor>(1);
  const Tensor* write_indices_tensor = context->Input<Tensor>(2);  // optional

  ORT_ENFORCE(past_cache != nullptr && update != nullptr,
              "TensorScatter: past_cache and update must not be null");

  const auto& cache_shape = past_cache->Shape();
  const auto& update_shape = update->Shape();
  const int ndim = static_cast<int>(cache_shape.NumDimensions());

  ORT_ENFORCE(ndim >= 2, "TensorScatter: past_cache must have at least 2 dimensions");
  ORT_ENFORCE(update_shape.NumDimensions() == cache_shape.NumDimensions(),
              "TensorScatter: past_cache and update must have the same number of dimensions");

  // Resolve axis (handles negative values).
  int axis = static_cast<int>(axis_);
  if (axis < 0) axis += ndim;
  ORT_ENFORCE(axis > 0 && axis < ndim,
              "TensorScatter: axis must be in [1, ndim-1] after normalization, got ", axis);

  // Validate shapes: all dimensions must match except the axis dimension.
  const int64_t batch_size = cache_shape[0];
  const int64_t max_sequence_length = cache_shape[axis];
  const int64_t sequence_length = update_shape[axis];

  ORT_ENFORCE(sequence_length <= max_sequence_length,
              "TensorScatter: update sequence_length (", sequence_length,
              ") exceeds max_sequence_length (", max_sequence_length, ")");

  for (int d = 0; d < ndim; ++d) {
    if (d != axis) {
      ORT_ENFORCE(cache_shape[d] == update_shape[d],
                  "TensorScatter: shape mismatch in dimension ", d,
                  ": past_cache=", cache_shape[d], " vs update=", update_shape[d]);
    }
  }

  // Validate write_indices if provided.
  const int64_t* write_indices = nullptr;
  if (write_indices_tensor != nullptr) {
    ORT_ENFORCE(write_indices_tensor->Shape().NumDimensions() == 1 &&
                    write_indices_tensor->Shape()[0] == batch_size,
                "TensorScatter: write_indices must have shape [batch_size]");
    write_indices = write_indices_tensor->Data<int64_t>();
  }

  // Allocate output with the same shape as past_cache.
  Tensor* present_cache = context->Output(0, cache_shape);

  // Step 1: Copy past_cache -> present_cache.
  const auto element_size = past_cache->DataType()->Size();
  const size_t total_bytes = SafeInt<size_t>(cache_shape.Size()) * element_size;
  const auto* src_raw = past_cache->DataRaw();
  auto* dst_raw = present_cache->MutableDataRaw();
  if (dst_raw != src_raw) {
    LOGS(context->Logger(), WARNING) << "TensorScatter: in-place optimization not activated, copying past_cache to present_cache ("
                                     << total_bytes << " bytes)";
    memcpy(dst_raw, src_raw, total_bytes);
  }

  // Step 2: Scatter the update into present_cache.
  //
  // Layout: (batch_size, D1, ..., D_{axis-1}, max_seq_len, D_{axis+1}, ..., D_{n-1})
  //
  // We decompose the tensor into:
  //   prefix_count = product of dims[0:axis]       (number of prefix slices)
  //   suffix_bytes = product of dims[axis+1:] * element_size  (bytes per single sequence position)
  //
  // For each prefix slice we determine batch_idx = prefix_linear_idx / prefix_stride_for_batch,
  // look up write_indices[batch_idx], and memcpy sequence_length suffix-sized chunks.

  int64_t prefix_count = 1;
  for (int d = 0; d < axis; ++d) {
    prefix_count *= cache_shape[d];
  }

  int64_t suffix_count = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    suffix_count *= cache_shape[d];
  }
  const size_t suffix_bytes = SafeInt<size_t>(suffix_count) * element_size;

  // prefix_stride_for_batch: number of prefix elements per batch element
  // e.g., shape (B, D1, ..., D_{axis-1}, ...) -> stride = D1 * ... * D_{axis-1}
  int64_t prefix_stride_for_batch = 1;
  for (int d = 1; d < axis; ++d) {
    prefix_stride_for_batch *= cache_shape[d];
  }

  const size_t cache_axis_stride = SafeInt<size_t>(max_sequence_length) * suffix_bytes;
  const size_t update_axis_stride = SafeInt<size_t>(sequence_length) * suffix_bytes;
  auto* dst_bytes = static_cast<uint8_t*>(dst_raw);
  const auto* update_raw = static_cast<const uint8_t*>(update->DataRaw());

  for (int64_t p = 0; p < prefix_count; ++p) {
    int64_t batch_idx = p / prefix_stride_for_batch;
    int64_t wi = (write_indices != nullptr) ? write_indices[batch_idx] : 0;
    ORT_ENFORCE(wi >= 0, "TensorScatter: write_indices[", batch_idx, "] = ", wi, " is negative");

    ptrdiff_t update_offset = static_cast<ptrdiff_t>(SafeInt<size_t>(p) * update_axis_stride);
    ptrdiff_t cache_offset = static_cast<ptrdiff_t>(SafeInt<size_t>(p) * cache_axis_stride);
    const uint8_t* update_base = update_raw + update_offset;
    uint8_t* cache_base = dst_bytes + cache_offset;

    if (!circular_) {
      ORT_ENFORCE(wi + sequence_length <= max_sequence_length,
                  "TensorScatter linear mode: write_indices[", batch_idx, "] + sequence_length (",
                  wi, " + ", sequence_length, ") exceeds max_sequence_length (", max_sequence_length, ")");
      // Single contiguous memcpy for the whole slice.
      ptrdiff_t wi_offset = static_cast<ptrdiff_t>(SafeInt<size_t>(wi) * suffix_bytes);
      size_t copy_len = SafeInt<size_t>(sequence_length) * suffix_bytes;
      memcpy(cache_base + wi_offset, update_base, copy_len);
    } else {
      // Circular: each sequence position wraps independently.
      for (int64_t s = 0; s < sequence_length; ++s) {
        int64_t cache_pos = (wi + s) % max_sequence_length;
        ptrdiff_t dst_off = static_cast<ptrdiff_t>(SafeInt<size_t>(cache_pos) * suffix_bytes);
        ptrdiff_t src_off = static_cast<ptrdiff_t>(SafeInt<size_t>(s) * suffix_bytes);
        memcpy(cache_base + dst_off, update_base + src_off, suffix_bytes);
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
