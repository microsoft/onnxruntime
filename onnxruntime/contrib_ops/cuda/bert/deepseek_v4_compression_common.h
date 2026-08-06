// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <string>

#include "core/common/narrow.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchDeepSeekV4CompressorKernel(
  cudaStream_t stream, T* entries, T* pending_kv_out, T* pending_gate_out,
  T* overlap_kv_out, T* overlap_gate_out, const T* current_kv, const T* current_gate,
  const T* past_pending_kv, const T* past_pending_gate, const T* past_overlap_kv,
  const T* past_overlap_gate, const T* position_bias, const T* norm_weight,
  const T* cos_cache, const T* sin_cache, const int64_t* position_ids,
  int batch_size, int sequence_length,
  int pending_token_count, int old_entry_count, int new_entry_count, int width,
  int head_size, int compress_rate, int rotary_dim, int cos_cache_width, int entry_capacity,
  float epsilon, bool is_overlap, bool fixed_mode, int max_threads_per_block);

namespace {

template <typename T>
struct CompressorState {
  using CudaT = typename ToCudaType<T>::MappedType;
  CudaT* entries{};
  int entry_count{};
  int pending_count{};
  int head_size{};
  bool rank4{};
};

template <typename T>
Status Project(const CudaKernel& kernel, OpKernelContext* context, const Tensor& input,
               const Tensor& weight, int rows, int input_width, int output_width,
               typename ToCudaType<T>::MappedType* output) {
  using CudaT = typename ToCudaType<T>::MappedType;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      kernel.GetCublasHandle(context), CUBLAS_OP_N, CUBLAS_OP_N,
      output_width, rows, input_width, &one,
      reinterpret_cast<const CudaT*>(weight.Data<T>()), output_width,
      reinterpret_cast<const CudaT*>(input.Data<T>()), input_width,
      &zero, output, output_width, kernel.GetDeviceProp(), kernel.UseTF32()));
    return Status::OK();
}

template <typename T>
Status RunCompressor(const CudaKernel& kernel, OpKernelContext* context,
                     const Tensor& hidden, const Tensor& cos_cache, const Tensor& sin_cache,
                     const Tensor& kv_weight, const Tensor& gate_weight,
                     const Tensor& position_bias, const Tensor& norm_weight,
                     const Tensor& past_pending_kv, const Tensor& past_pending_gate,
                     const Tensor& past_entries, const Tensor* past_overlap_kv,
                     const Tensor* past_overlap_gate, int64_t compress_rate,
                     int64_t rotary_dim, float epsilon, int entries_output_index,
                     int pending_kv_output_index, int pending_gate_output_index,
                     int overlap_kv_output_index, int overlap_gate_output_index,
                     CompressorState<T>& state, int64_t entry_capacity = 0,
                     const Tensor* position_ids = nullptr) {
  using CudaT = typename ToCudaType<T>::MappedType;
  const auto& hidden_shape = hidden.Shape();
  ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 3, "hidden_states must have rank 3.");
  const int64_t batch = hidden_shape[0];
  const int64_t sequence = hidden_shape[1];
  const int64_t hidden_size = hidden_shape[2];
  const bool is_overlap = past_overlap_kv != nullptr;
  ORT_RETURN_IF_NOT(kv_weight.Shape().NumDimensions() == 2 && kv_weight.Shape()[0] == hidden_size,
                    "kv_weight has an incompatible shape.");
  const int64_t width = kv_weight.Shape()[1];
  const int64_t head_size = is_overlap ? width / 2 : width;
  ORT_RETURN_IF(is_overlap && width % 2 != 0, "overlap compressor width must be even.");
  ORT_RETURN_IF_NOT(gate_weight.Shape() == kv_weight.Shape() &&
                        position_bias.Shape() == TensorShape({compress_rate, width}) &&
                        norm_weight.Shape() == TensorShape({head_size}),
                    "compressor projection, position bias, or norm shape mismatch.");
  ORT_RETURN_IF_NOT(cos_cache.Shape() == sin_cache.Shape() && cos_cache.Shape().NumDimensions() == 2 &&
                        rotary_dim <= head_size && rotary_dim % 2 == 0 &&
                        cos_cache.Shape()[1] * 2 >= rotary_dim,
                    "compressor RoPE inputs are incompatible.");
  ORT_RETURN_IF_NOT(past_pending_kv.Shape().NumDimensions() == 3 && past_pending_kv.Shape()[0] == batch &&
                        past_pending_kv.Shape()[2] == width && past_pending_gate.Shape() == past_pending_kv.Shape(),
                    "compressor pending state shape mismatch.");
  const bool fixed_mode = entry_capacity > 0;
  const int64_t pending_capacity = past_pending_kv.Shape()[1];
  ORT_RETURN_IF_NOT(fixed_mode ? pending_capacity == compress_rate - 1 : pending_capacity < compress_rate,
                    "pending state must be shorter than compress_rate, with capacity compress_rate - 1 in fixed mode.");
  const auto& entries_shape = past_entries.Shape();
  ORT_RETURN_IF_NOT(entries_shape.NumDimensions() == 3 || entries_shape.NumDimensions() == 4,
                    "entries state must have rank 3 or 4.");
  const bool rank4 = entries_shape.NumDimensions() == 4;
  ORT_RETURN_IF(rank4 && entries_shape[1] != 1, "rank-4 entries must have singleton head dimension.");
  const int64_t entries_extent = rank4 ? entries_shape[2] : entries_shape[1];
  ORT_RETURN_IF_NOT(entries_shape[0] == batch && entries_shape[entries_shape.NumDimensions() - 1] == head_size,
                    "entries state shape mismatch.");
  ORT_RETURN_IF(fixed_mode && (entries_extent != entry_capacity || position_ids == nullptr),
                "fixed-mode entries shape or position_ids is invalid.");
  if (is_overlap) {
    ORT_RETURN_IF_NOT(past_overlap_kv->Shape() == TensorShape({batch, compress_rate, head_size}) &&
                          past_overlap_gate->Shape() == past_overlap_kv->Shape(),
                      "overlap state shape mismatch.");
  }

  const int64_t pending_count = fixed_mode ? 0 : pending_capacity;
  const int64_t old_entry_count = fixed_mode ? 0 : entries_extent;
  const int64_t total_count = pending_count + sequence;
  const int64_t new_entry_count = fixed_mode
                                      ? (sequence + compress_rate - 1) / compress_rate
                                      : total_count / compress_rate;
  const int64_t output_pending_count = fixed_mode ? pending_capacity : total_count % compress_rate;
  const int64_t output_entry_count = fixed_mode ? entry_capacity : old_entry_count + new_entry_count;
  ORT_RETURN_IF(!fixed_mode && new_entry_count > 0 && (output_entry_count - 1) * compress_rate >= cos_cache.Shape()[0],
                "compressed entry position is outside the RoPE cache.");
  ORT_RETURN_IF(fixed_mode && entry_capacity * compress_rate > cos_cache.Shape()[0],
                "fixed entry capacity exceeds the RoPE cache.");
  const TensorShape output_entries_shape = rank4
                                               ? TensorShape({batch, 1, output_entry_count, head_size})
                                               : TensorShape({batch, output_entry_count, head_size});
  Tensor* entries_output = context->Output(entries_output_index, output_entries_shape);
  Tensor* pending_kv_output = context->Output(
      pending_kv_output_index, TensorShape({batch, output_pending_count, width}));
  Tensor* pending_gate_output = context->Output(
      pending_gate_output_index, TensorShape({batch, output_pending_count, width}));
  Tensor* overlap_kv_output = is_overlap
                                  ? context->Output(overlap_kv_output_index, past_overlap_kv->Shape())
                                  : nullptr;
  Tensor* overlap_gate_output = is_overlap
                                    ? context->Output(overlap_gate_output_index, past_overlap_gate->Shape())
                                    : nullptr;
  cudaStream_t stream = kernel.Stream(context);
  CudaT* entries = reinterpret_cast<CudaT*>(entries_output->MutableData<T>());
  const CudaT* old_entries = reinterpret_cast<const CudaT*>(past_entries.Data<T>());
  if (entries != old_entries) {
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          entries + batch_index * output_entry_count * head_size,
          old_entries + batch_index * entries_extent * head_size,
          static_cast<size_t>(entries_extent * head_size) * sizeof(CudaT),
          cudaMemcpyDeviceToDevice, stream));
    }
  }
  CudaT* overlap_kv = overlap_kv_output
                          ? reinterpret_cast<CudaT*>(overlap_kv_output->MutableData<T>())
                          : nullptr;
  CudaT* overlap_gate = overlap_gate_output
                            ? reinterpret_cast<CudaT*>(overlap_gate_output->MutableData<T>())
                            : nullptr;
  if (is_overlap && !fixed_mode && new_entry_count == 0) {
    const size_t overlap_bytes = static_cast<size_t>(past_overlap_kv->Shape().Size()) * sizeof(CudaT);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(overlap_kv, past_overlap_kv->Data<T>(), overlap_bytes,
                                         cudaMemcpyDeviceToDevice, stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(overlap_gate, past_overlap_gate->Data<T>(), overlap_bytes,
                                         cudaMemcpyDeviceToDevice, stream));
  }
  auto current_kv = kernel.GetScratchBuffer<CudaT>(static_cast<size_t>(batch * sequence * width), stream);
  auto current_gate = kernel.GetScratchBuffer<CudaT>(static_cast<size_t>(batch * sequence * width), stream);
  auto pending_kv_staging = fixed_mode
                                ? kernel.GetScratchBuffer<CudaT>(static_cast<size_t>(past_pending_kv.Shape().Size()), stream)
                                : IAllocatorUniquePtr<CudaT>{};
  auto pending_gate_staging = fixed_mode
                                  ? kernel.GetScratchBuffer<CudaT>(static_cast<size_t>(past_pending_gate.Shape().Size()), stream)
                                  : IAllocatorUniquePtr<CudaT>{};
  auto overlap_kv_staging = fixed_mode && is_overlap
                                ? kernel.GetScratchBuffer<CudaT>(static_cast<size_t>(past_overlap_kv->Shape().Size()), stream)
                                : IAllocatorUniquePtr<CudaT>{};
  auto overlap_gate_staging = fixed_mode && is_overlap
                                  ? kernel.GetScratchBuffer<CudaT>(static_cast<size_t>(past_overlap_gate->Shape().Size()), stream)
                                  : IAllocatorUniquePtr<CudaT>{};
  if (fixed_mode) {
    const size_t pending_bytes = static_cast<size_t>(past_pending_kv.Shape().Size()) * sizeof(CudaT);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(pending_kv_staging.get(), past_pending_kv.Data<T>(), pending_bytes,
                                         cudaMemcpyDeviceToDevice, stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(pending_gate_staging.get(), past_pending_gate.Data<T>(), pending_bytes,
                                         cudaMemcpyDeviceToDevice, stream));
    if (is_overlap) {
      const size_t overlap_bytes = static_cast<size_t>(past_overlap_kv->Shape().Size()) * sizeof(CudaT);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(overlap_kv_staging.get(), past_overlap_kv->Data<T>(), overlap_bytes,
                                           cudaMemcpyDeviceToDevice, stream));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(overlap_gate_staging.get(), past_overlap_gate->Data<T>(), overlap_bytes,
                                           cudaMemcpyDeviceToDevice, stream));
    }
  }
  ORT_RETURN_IF_ERROR(Project<T>(kernel, context, hidden, kv_weight,
                                 narrow<int>(batch * sequence), narrow<int>(hidden_size),
                                 narrow<int>(width), current_kv.get()));
  ORT_RETURN_IF_ERROR(Project<T>(kernel, context, hidden, gate_weight,
                                 narrow<int>(batch * sequence), narrow<int>(hidden_size),
                                 narrow<int>(width), current_gate.get()));
  ORT_RETURN_IF_ERROR(LaunchDeepSeekV4CompressorKernel<CudaT>(
      stream, entries,
      reinterpret_cast<CudaT*>(pending_kv_output->MutableData<T>()),
      reinterpret_cast<CudaT*>(pending_gate_output->MutableData<T>()), overlap_kv, overlap_gate,
      current_kv.get(), current_gate.get(),
      fixed_mode ? pending_kv_staging.get() : reinterpret_cast<const CudaT*>(past_pending_kv.Data<T>()),
      fixed_mode ? pending_gate_staging.get() : reinterpret_cast<const CudaT*>(past_pending_gate.Data<T>()),
      is_overlap ? (fixed_mode ? overlap_kv_staging.get() : reinterpret_cast<const CudaT*>(past_overlap_kv->Data<T>())) : nullptr,
      is_overlap ? (fixed_mode ? overlap_gate_staging.get() : reinterpret_cast<const CudaT*>(past_overlap_gate->Data<T>())) : nullptr,
      reinterpret_cast<const CudaT*>(position_bias.Data<T>()),
      reinterpret_cast<const CudaT*>(norm_weight.Data<T>()),
      reinterpret_cast<const CudaT*>(cos_cache.Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache.Data<T>()),
      fixed_mode ? position_ids->Data<int64_t>() : nullptr, narrow<int>(batch), narrow<int>(sequence),
      narrow<int>(pending_count), narrow<int>(old_entry_count), narrow<int>(new_entry_count),
      narrow<int>(width), narrow<int>(head_size), narrow<int>(compress_rate), narrow<int>(rotary_dim),
      narrow<int>(cos_cache.Shape()[1]), narrow<int>(output_entry_count), epsilon, is_overlap,
      fixed_mode, kernel.GetDeviceProp().maxThreadsPerBlock));
  state.entries = entries;
  state.entry_count = narrow<int>(output_entry_count);
  state.pending_count = narrow<int>(output_pending_count);
  state.head_size = narrow<int>(head_size);
  state.rank4 = rank4;
  return Status::OK();
}

template <typename T>
Status ProjectTransposed(const CudaKernel& kernel, OpKernelContext* context, const Tensor& input,
                         const Tensor& weight, int rows, int input_width, int output_width,
                         typename ToCudaType<T>::MappedType* output) {
  using CudaT = typename ToCudaType<T>::MappedType;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      kernel.GetCublasHandle(context), CUBLAS_OP_T, CUBLAS_OP_N,
      output_width, rows, input_width, &one,
      reinterpret_cast<const CudaT*>(weight.Data<T>()), input_width,
      reinterpret_cast<const CudaT*>(input.Data<T>()), input_width,
      &zero, output, output_width, kernel.GetDeviceProp(), kernel.UseTF32()));
  return Status::OK();
}

}  // namespace
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
