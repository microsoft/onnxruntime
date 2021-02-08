// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_softmax.h"

#include <limits>
#include <algorithm>

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/math/binary_elementwise_ops_impl_functors.cuh"
#include "core/providers/cuda/math/softmax_impl.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Duplicated softmax_impl.cu here
// So far attempt to use shared kernel with additional template resulted in lost performance

// Note: The intended case for 'input_bias' is the input sequence mask for transformer models
// As an additive mask, it should be zero for preserved tokens and -infty for tokens to screen
// The mask will broadcast from [batch_size, 1, 1, seq_len] to input [batch_size, num_heads, seq_len, seq_len]
// Here element_count = seq_len and bias_broadcast_size_per_batch = num_heads * seq_len

// The softmax + additive mask fusion follows NVIDIA apex's additive_masked_softmax_warp_forward
// see https://github.com/NVIDIA/apex/blob/4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a/apex/contrib/csrc/multihead_attn/softmax.h

template <typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void BiasSoftmaxWarpForward(
    output_t* output,
    const input_t* input,
    const input_t* input_bias,
    int element_count,
    int batch_count,
    int batch_stride,
    int bias_broadcast_count_per_batch) {
  // "WARP" refers to cooperative threads and might not equal 32 threads of GPU warp
  // thread block is (WARP_SIZE, 128/WARP_SIZE)
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = next_power_of_two < GPU_WARP_SIZE ? next_power_of_two : GPU_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  // each "WARP" (<=32) processes WARP_BATCH(one of {1,2}) batches
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  // last warp may have fewer batches
  int local_batches = batch_count - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // thread will process elements (local_index + n * warp_size) within batch
  int local_idx = threadIdx.x;

  // push input, input_bias output pointers to batch we need to process
  input += first_batch * batch_stride + local_idx;
  output += first_batch * batch_stride + local_idx;

  // load from global memory and apply bias (likely an additive mask)
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    // the bias has assumed shape [batch_size, element_count]
    // .. and needs to broadcast to [batch_size, broadcast_size, element_count]
    int bias_offset = (first_batch + i) / bias_broadcast_count_per_batch * batch_stride + local_idx;

    int batch_element_count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = (acc_t)input[i * element_count + it * WARP_SIZE] + (acc_t)input_bias[bias_offset + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // find maximum value within batch for numerical stability
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  // normalization factor Z = Sum[ exp(element_i), for element_i in batch ]
  acc_t sum[WARP_BATCH]{acc_t(0.0)};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = std::exp((acc_t)(elements[i][it] - max_value[i]));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

// write back normalized value = exp(element_i)/Z to global memory
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches)
      break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        output[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
      } else {
        break;
      }
    }
  }
}

template <typename T>
void DispatchBiasSoftmaxForwardImpl(
    cudaStream_t stream,
    Tensor* output_tensor,
    const Tensor* input_tensor,
    const Tensor* input_bias_tensor,
    int element_count,
    int batch_count,
    int batch_stride,
    int bias_broadcast_size_per_batch) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef CudaT input_t;
  typedef CudaT output_t;
  typedef AccumulationType_t<CudaT> acc_t;

  const auto* input = reinterpret_cast<const CudaT*>(input_tensor->template Data<T>());
  const auto* input_bias = reinterpret_cast<const CudaT*>(input_bias_tensor->template Data<T>());
  auto* output = reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>());

  if (element_count == 0)
    return;

  int log2_elements = log2_ceil(element_count);
  const int next_power_of_two = 1 << log2_elements;

  // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
  int warp_size = std::min(next_power_of_two, GPU_WARP_SIZE);

  // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
  int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

  // use 128 threads per block to maximize gpu utilization
  constexpr int threads_per_block = 128;

  int warps_per_block = (threads_per_block / warp_size);
  int batches_per_block = warps_per_block * batches_per_warp;
  int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
  dim3 threads(warp_size, warps_per_block, 1);

  // Launch code would be more elegant if C++ supported FOR CONSTEXPR
  switch (log2_elements) {
    case 0:  // 1
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 0>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 1:  // 2
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 1>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 2:  // 4
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 2>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 3:  // 8
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 3>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 4:  // 16
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 4>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 5:  // 32
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 5>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 6:  // 64
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 6>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 7:  // 128
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 7>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 8:  // 256
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 8>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 9:  // 512
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 9>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    case 10:  // 1024
      BiasSoftmaxWarpForward<input_t, output_t, acc_t, 10>
          <<<blocks, threads, 0, stream>>>(output, input, input_bias, element_count, batch_count, batch_stride, bias_broadcast_size_per_batch);
      break;
    default:
      break;
  }
}

#define SPECIALIZED_BIAS_SOFTMAX_IMPL(T)           \
  template void DispatchBiasSoftmaxForwardImpl<T>( \
      cudaStream_t stream,                         \
      Tensor * output_tensor,                      \
      const Tensor* input_tensor,                  \
      const Tensor* input_bias_tensor,             \
      int element_count,                           \
      int batch_count,                             \
      int batch_stride,                            \
      int bias_broadcast_size_per_batch);

SPECIALIZED_BIAS_SOFTMAX_IMPL(double)
SPECIALIZED_BIAS_SOFTMAX_IMPL(float)
SPECIALIZED_BIAS_SOFTMAX_IMPL(MLFloat16)

// For large element count we fall back to explicit Add kernel + CUDA DNN library
// note: This is an unhappy path! There is no performance benefit for the fusion.
template <typename T>
void DispatchBiasSoftMaxForwardViaDnnLibraryImpl(
    cudaStream_t stream,
    cudnnHandle_t cudaDnnHandle,
    int element_count,
    int batch_count,
    int broadcast_axis,
    int softmax_axis,
    const onnxruntime::TensorShape& X_shape,
    const onnxruntime::Tensor* X,
    const onnxruntime::TensorShape& B_shape,
    const onnxruntime::Tensor* B,
    onnxruntime::Tensor* Y) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const auto* X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  const auto* B_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
  auto* Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  // binary elementise kernel requires input pitches
  TArray<int64_t> lhs_padded_strides(static_cast<int>(X_shape.NumDimensions()));
  int64_t lhs_pitch = 1, rhs_pitch = 1;
  for (int i = -1; i >= -(int)X_shape.NumDimensions(); i--) {
    size_t positive_i = X_shape.NumDimensions() + i;
    lhs_padded_strides[static_cast<int>(positive_i)] = lhs_pitch;
    lhs_pitch *= X_shape[positive_i];
  }

  // set pitches for bias so it broadcasts along relevant dimensions
  TArray<int64_t> rhs_padded_strides(static_cast<int>(X_shape.NumDimensions()));
  for (int i = -1; i >= -(int)X_shape.NumDimensions(); i--) {
    size_t positive_ix = X_shape.NumDimensions() + i;
    size_t positive_ib = B_shape.NumDimensions() + i;
    if (broadcast_axis <= positive_ix && positive_ix < softmax_axis) {
      rhs_padded_strides[static_cast<int>(positive_ix)] = 0;
      continue;
    }
    rhs_padded_strides[static_cast<int>(positive_ix)] = rhs_pitch;
    rhs_pitch *= B_shape[positive_ib];
  }

  TArray<fast_divmod> fdm_output_strides(static_cast<int>(X_shape.NumDimensions()));
  //TODO: fast_divmod only supports int32
  for (int i = 0; i < fdm_output_strides.Size(); i++)
    fdm_output_strides[i] = fast_divmod(static_cast<int>(lhs_padded_strides[i]));
  fast_divmod fdm_H, fdm_C;

  // invoke elementwise add with broadcast kernel
  ::onnxruntime::cuda::BinaryElementWiseImpl(
      stream,
      (int32_t)X_shape.NumDimensions(),
      &lhs_padded_strides,
      X_data,
      &rhs_padded_strides,
      B_data,
      &fdm_output_strides,
      fdm_H,
      fdm_C,
      Y_data,
      OP_Add<CudaT, CudaT, CudaT>(),
      (size_t)X_shape.Size());

  // invoke cuda DNN library for Y = softmax(X)
  std::vector<int64_t> dims({batch_count, 1, 1, element_count});
  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  CudnnTensor input_tensor, output_tensor;
  input_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>());
  output_tensor.Set(dims, CudnnTensor::GetDataType<CudaT>());
  cudnnSoftmaxForward(
      cudaDnnHandle,
      CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_INSTANCE,
      &alpha,
      input_tensor,
      Y_data,
      &beta,
      output_tensor,
      Y_data);
}

#define SPECIALIZED_BIAS_SOFTMAX_IMPL_VIA_DNN(T)                \
  template void DispatchBiasSoftMaxForwardViaDnnLibraryImpl<T>( \
      cudaStream_t stream,                                      \
      cudnnHandle_t cudaDnnHandle,                              \
      int element_count,                                        \
      int batch_count,                                          \
      int broadcast_axis,                                       \
      int softmax_axis,                                         \
      const onnxruntime::TensorShape& X_shape,                  \
      const Tensor* X_data,                                     \
      const onnxruntime::TensorShape& B_shape,                  \
      const Tensor* B_data,                                     \
      Tensor* Y_data);

SPECIALIZED_BIAS_SOFTMAX_IMPL_VIA_DNN(double)
SPECIALIZED_BIAS_SOFTMAX_IMPL_VIA_DNN(float)
SPECIALIZED_BIAS_SOFTMAX_IMPL_VIA_DNN(MLFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
