// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/llm/tensorscatter.h"
#include "core/providers/cuda/llm/tensorscatter_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    TensorScatter,
    kOnnxDomain,
    24,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    TensorScatter);

TensorScatter::TensorScatter(const OpKernelInfo& info) : CudaKernel(info) {
  axis_ = info.GetAttrOrDefault<int64_t>("axis", -2);
  std::string mode = info.GetAttrOrDefault<std::string>("mode", "linear");
  ORT_ENFORCE(mode == "linear" || mode == "circular",
              "TensorScatter: mode must be 'linear' or 'circular', got '", mode, "'");
  circular_ = (mode == "circular");
}

Status TensorScatter::ComputeInternal(OpKernelContext* context) const {
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

    // Copy write_indices to host for validation (batch_size elements, negligible overhead).
    std::vector<int64_t> host_write_indices(static_cast<size_t>(batch_size));
    CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(host_write_indices.data(), write_indices,
                        static_cast<size_t>(batch_size) * sizeof(int64_t),
                        cudaMemcpyDeviceToHost, Stream(context)));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(Stream(context)));

    for (int64_t b = 0; b < batch_size; ++b) {
      int64_t wi = host_write_indices[static_cast<size_t>(b)];
      ORT_ENFORCE(wi >= 0, "TensorScatter: write_indices[", b, "] = ", wi, " is negative");
      if (!circular_) {
        ORT_ENFORCE(wi + sequence_length <= max_sequence_length,
                    "TensorScatter linear mode: write_indices[", b, "] + sequence_length (",
                    wi, " + ", sequence_length, ") exceeds max_sequence_length (", max_sequence_length, ")");
      }
    }
  }

  // Allocate output with the same shape as past_cache.
  Tensor* present_cache = context->Output(0, cache_shape);

  // Step 1: Copy past_cache -> present_cache.
  const void* src_raw = past_cache->DataRaw();
  void* dst_raw = present_cache->MutableDataRaw();
  if (dst_raw != src_raw) {
    CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(dst_raw, src_raw, past_cache->SizeInBytes(),
                        cudaMemcpyDeviceToDevice, Stream(context)));
  }

  // Bail out early if nothing to scatter.
  if (sequence_length == 0) {
    return Status::OK();
  }

  // Step 2: Scatter the update into present_cache.
  const size_t element_size = past_cache->DataType()->Size();

  int64_t prefix_count = 1;
  for (int d = 0; d < axis; ++d) {
    prefix_count *= cache_shape[d];
  }

  int64_t suffix_count = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    suffix_count *= cache_shape[d];
  }

  int64_t prefix_stride_for_batch = 1;
  for (int d = 1; d < axis; ++d) {
    prefix_stride_for_batch *= cache_shape[d];
  }

  return TensorScatterImpl(
      Stream(context),
      dst_raw,
      update->DataRaw(),
      write_indices,
      element_size,
      prefix_count,
      prefix_stride_for_batch,
      max_sequence_length,
      sequence_length,
      suffix_count,
      circular_);
}

}  // namespace cuda
}  // namespace onnxruntime
