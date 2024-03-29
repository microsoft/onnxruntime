// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "orttraining/training_ops/cuda/transformer/variable_length_attention_impl.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
// #include "core/providers/cuda/math/softmax_warpwise_impl.cuh"

#include "core/providers/cuda/math/softmax.h"
namespace onnxruntime {
namespace cuda {

// // Grid: (S, B)
// // Block: 256
// // For unfused PackedAttention
// //     Input: Tx3xNxH
// //     Output: 3xBxNxSxH
// // Where:
// // T is token_count
// // B is batch_size
// // S is sequence_length
// // N is num_heads
// // H is head_size
// template <typename T>
// __global__ void AddBiasTransposeQKVPacked(
//     const T* input,
//     const T* biases,
//     int32_t N,
//     int32_t H_QK,
//     int32_t H_V,
//     T* q,
//     T* k,
//     T* v,
//     const int32_t* token_offset,
//     int32_t token_count) {
//   int s = blockIdx.x;
//   int b = blockIdx.y;

//   int S = gridDim.x;

//   const int packing_token_idx = b * S + s;
//   const int padding_token_idx = token_offset[packing_token_idx];
//   b = padding_token_idx / S;
//   s = padding_token_idx - b * S;

//   input += packing_token_idx * N * (H_QK + H_QK + H_V);
//   int k_offset = N * H_QK;
//   int v_offset = N * H_QK + N * H_QK;
//   q += (b * N * S + s) * H_QK;
//   k += (b * N * S + s) * H_QK;
//   v += (b * N * S + s) * H_V;

//   if (packing_token_idx < token_count) {
//     for (int i = threadIdx.x; i < N * H_QK; i += blockDim.x) {
//       int h = i % H_QK;
//       int n = i / H_QK;
//       q[n * S * H_QK + h] = input[i] + biases[i];
//       k[n * S * H_QK + h] = input[i + k_offset] + biases[i + k_offset];
//     }

//     for (int i = threadIdx.x; i < N * H_V; i += blockDim.x) {
//       int h = i % H_V;
//       int n = i / H_V;
//       v[n * S * H_V + h] = input[i + v_offset] + biases[i + v_offset];
//     }
//   } else {
//     for (int i = threadIdx.x; i < N * H_QK; i += blockDim.x) {
//       int h = i % H_QK;
//       int n = i / H_QK;
//       q[n * S * H_QK + h] = biases[i];
//       k[n * S * H_QK + h] = biases[i + k_offset];
//     }

//     for (int i = threadIdx.x; i < N * H_V; i += blockDim.x) {
//       int h = i % H_V;
//       int n = i / H_V;
//       v[n * S * H_V + h] = biases[i + v_offset];
//     }
//   }
// }

/**
 * Input: [[seq1, head, hidden_size_per_head],
 *         [seq2, head, hidden_size_per_head],
 *         ...
 *         [seqN, head, hidden_size_per_head]]
 * Attr: perms= [0, 2, 3, 1]
 * Output: [[hidden_size_per_head, seq1, head],
 *          [hidden_size_per_head, seq2, head],
 *          ...
 *          [hidden_size_per_head, seqN, head]]
 */

/**
 * Input: [[seq1, head, hidden_size_per_head],
 *         [seq2, head, hidden_size_per_head],
 *         ...
 *         [seqN, head, hidden_size_per_head]]
 * Attr: perms= [0, 2, 1, 3]
 * Output: [[head, seq1, hidden_size_per_head],
 *          [head, seq2, hidden_size_per_head],
 *          ...
 *          [head, seqN, hidden_size_per_head]]
 */

template <typename T>
__global__ void GroupTransposeKernelImpl(const int64_t* cum_seq_length,
                                         int variant_axis_in_output,
                                         int variant_axis_on_output,
                                         const TArray<int64_t> input_shape,
                                         const TArray<int64_t> output_shape,
                                         int64_t factor_for_fixed_dims,
                                         const TArray<int> perms,
                                         const T* input_data,
                                         T* output_data,
                                         CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  CUDA_LONG output_index = id;
  int batch_count = input_shape[0];
  int flatten_seq_index = CeilDiv(output_index, factor_for_fixed_dims);
  int batch_index = 0;
#pragma unroll
  for (batch_index = 0; batch_index < batch_count; ++batch_index) {
    if (flatten_seq_index < cum_seq_length[batch_index + 1]) {
      break;
    }
  }

  // cum_seq_length example:
  // [0, 3, 6, 9, 12], batch_count = 4, each element in the array represent the
  // accumulated sequence lengths before the current batch (exclusive). The first element should always be 0.

  // We assume batch dim is not transposed in the perm.
  int64_t batch_start_offset = cum_seq_length[batch_index] * factor_for_fixed_dims;
  int64_t token_index_in_the_batch = flatten_seq_index - cum_seq_length[batch_index];
  int64_t seq_length_for_the_batch = cum_seq_length[batch_index + 1] - cum_seq_length[batch_index];

  int offset_within_the_batch = token_index_in_the_batch * factor_for_fixed_dims;
  TArray<int> input_strides_in_the_batch(3);
  TArray<onnxruntime::cuda::fast_divmod> output_strides_in_the_batch(3);

  // Build output and input stride dynamically, this should be fast because they operates on register.

  int input_acc_elem_stride = 1;
  int output_acc_elem_stride = 1;
#pragma unroll
  for (auto dim = 3; dim > 0; ++dim) {
    if (dim == variant_axis_in_output) {
      input_acc_elem_stride *= seq_length_for_the_batch;
    } else {
      input_acc_elem_stride *= input_shape[dim];
    }

    if (dim == variant_axis_on_output) {
      output_acc_elem_stride *= seq_length_for_the_batch;
    } else {
      output_acc_elem_stride *= output_shape[dim];
    }

    input_strides_in_the_batch[dim - 1] = input_acc_elem_stride;
    output_strides_in_the_batch[dim - 1] = onnxruntime::cuda::fast_divmod(output_acc_elem_stride);
  }

  CUDA_LONG input_index = batch_start_offset;
  int remain = offset_within_the_batch;
  int dim = 0;
#pragma unroll
  for (auto dim = 0; dim < 3; ++dim) {
    output_strides_in_the_batch[dim].divmod(remain, dim, remain);
    input_index += dim * input_strides_in_the_batch[perms[dim + 1] - 1];
  }

  output_data[id] = input_data[input_index];
}

template <typename T>
Status LaunchGroupTranspose(cudaStream_t stream,
                            size_t element_size,
                            const int64_t* cum_seq_length,
                            int variant_axis_in_output,
                            int variant_axis_on_output,
                            const TArray<int64_t> input_shape,
                            const TArray<int64_t> output_shape,
                            int64_t factor_for_fixed_dims,
                            const TArray<int> reverse_perms,
                            const T* input_data,
                            T* output_data,
                            size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (element_size) {
    case sizeof(int16_t):
      GroupTransposeKernelImpl<int16_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          cum_seq_length, variant_axis_in_output, variant_axis_on_output,
          input_shape, output_shape, factor_for_fixed_dims, reverse_perms,
          reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
          static_cast<CUDA_LONG>(N));
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }

  return Status::OK();
}

// template <typename T>
// __global__ void TransposeKernel(int32_t shape_rank, const TArray<int64_t> input_strides,
//                                 const T* input_data, const TArray<fast_divmod> output_strides, T* output_data, CUDA_LONG N) {
//   CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
//   CUDA_LONG input_index = 0;
//   CUDA_LONG output_index = id;

// #pragma unroll
//   for (auto dim = 0; dim < input_strides.Capacity(); ++dim) {
//     if (dim >= shape_rank) {
//       break;
//     }
//     int out_coord, r;
//     output_strides[dim].divmod(output_index, out_coord, r);
//     output_index = r;
//     input_index += input_strides[dim] * out_coord;
//   }
//   output_data[id] = input_data[input_index];
// }

template <typename T>
__global__ void GroupTransposeKernelImpl(const int64_t* cum_seq_length,
                                         int variant_axis_in_output,
                                         //  int variant_axis_on_output,
                                         const TArray<int64_t> input_shape,
                                         //  const TArray<int64_t> output_shape,
                                         int64_t factor_for_fixed_dims,
                                         const TArray<int> perms,
                                         const T* input_data,
                                         const TArray<fast_divmod> output_strides,
                                         T* output_data,
                                         CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  CUDA_LONG output_index = id;
  //   int batch_count = input_shape[0];
  //   int flatten_seq_index = CeilDiv(output_index, factor_for_fixed_dims);
  //   int batch_index = 0;
  // #pragma unroll
  //   for (batch_index = 0; batch_index < batch_count; ++batch_index) {
  //     if (flatten_seq_index < cum_seq_length[batch_index + 1]) {
  //       break;
  //     }
  //   }
  int out_coord, r;
  output_strides[0].divmod(output_index, out_coord, r);
  int batch_index = out_coord;
  int64_t token_index_in_the_batch = CeilDiv(r, factor_for_fixed_dims);
  if (token_index_in_the_batch >= cum_seq_length[batch_index + 1] - cum_seq_length[batch_index]) {
    output_data[id] = 0;
    return;
  }

  // cum_seq_length example:
  // [0, 3, 6, 9, 12], batch_count = 4, each element in the array represent the
  // accumulated sequence lengths before the current batch (exclusive). The first element should always be 0.

  // We assume batch dim is not transposed in the perm.
  int64_t batch_start_offset = cum_seq_length[batch_index] * factor_for_fixed_dims;
  // int64_t token_index_in_the_batch = flatten_seq_index - cum_seq_length[batch_index];
  int64_t seq_length_for_the_batch = cum_seq_length[batch_index + 1] - cum_seq_length[batch_index];

  // int offset_within_the_batch = token_index_in_the_batch * factor_for_fixed_dims;
  TArray<int> input_strides_in_the_batch(3);
  // TArray<onnxruntime::cuda::fast_divmod> output_strides_in_the_batch(3);

  // Build output and input stride dynamically, this should be fast because they operate on the register.

  int input_acc_elem_stride = 1;
  // int output_acc_elem_stride = 1;
#pragma unroll
  for (auto dim = 3; dim > 0; ++dim) {
    if (dim == variant_axis_in_output) {
      input_acc_elem_stride *= seq_length_for_the_batch;
    } else {
      input_acc_elem_stride *= input_shape[dim];
    }

    // if (dim == variant_axis_on_output) {
    //   output_acc_elem_stride *= seq_length_for_the_batch;
    // } else {
    //   output_acc_elem_stride *= output_shape[dim];
    // }

    input_strides_in_the_batch[dim - 1] = input_acc_elem_stride;
    // output_strides_in_the_batch[dim - 1] = onnxruntime::cuda::fast_divmod(output_acc_elem_stride);
  }

  CUDA_LONG input_index = batch_start_offset;
  // int remain = offset_within_the_batch;
  int dim_index = 0;
#pragma unroll
  for (auto dim = 1; dim <= 3; ++dim) {
    output_strides[dim].divmod(r, dim_index, r);
    input_index += dim_index * input_strides_in_the_batch[perms[dim - 1] - 1];
  }

  output_data[id] = input_data[input_index];
}

// template <typename T>
// Status LaunchGroupTranspose(cudaStream_t stream,
//                             size_t element_size,
//                             const int64_t* cum_seq_length,
//                             int variant_axis_in_output,
//                             // int variant_axis_on_output,
//                             const TArray<int64_t> input_shape,
//                             // const TArray<int64_t> output_shape,
//                             int64_t factor_for_fixed_dims,
//                             const TArray<int> reverse_perms,
//                             const T* input_data,
//                             const TArray<fast_divmod> output_strides,
//                             T* output_data,
//                             size_t N) {
//   int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
//   switch (element_size) {
//     case sizeof(int16_t):
//       GroupTransposeKernelImpl<int16_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
//           cum_seq_length, variant_axis_in_output,
//           input_shape, factor_for_fixed_dims, reverse_perms,
//           reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
//           output_strides,
//           reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
//           static_cast<CUDA_LONG>(N));
//       break;
//     default:
//       return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
//                              element_size);
//   }

//   return Status::OK();
// }

template <typename T, typename TOut, bool is_log_softmax>
Status SoftMaxVarLengthComputeHelper(
    Stream* stream,
    const T* X,
    const int64_t max_seq_lengh,
    const int64_t* cu_seqlens,
    const int64_t seqlen_count,
    const int64_t head_count,
    TOut* Y) {
  typedef typename ToCudaType<T>::MappedType CudaT_IN;
  typedef typename ToCudaType<TOut>::MappedType CudaT_OUT;
  typedef typename ToCudaType<T>::MappedType CudaT_ACCUM;

  // const int64_t axis = 3;
  // int64_t N = input_shape.SizeToDimension(axis);
  // int64_t D = input_shape.SizeFromDimension(axis);
  auto Y_data = reinterpret_cast<CudaT_OUT*>(Y);
  auto X_data = reinterpret_cast<const CudaT_IN*>(X);

  const int64_t batch_count = cu_seqlens[seqlen_count + 1] * head_count;

  if (max_seq_lengh <= 1024 && max_seq_lengh * sizeof(T) <= 4096) {
    return dispatch_warpwise_softmax_varlength_forward<
        CudaT_IN, CudaT_OUT, AccumulationType_t<CudaT_ACCUM>, is_log_softmax>(
        stream, Y_data, X_data, gsl::narrow_cast<int>(max_seq_lengh), gsl::narrow_cast<int>(batch_count),
        cu_seqlens, gsl::narrow_cast<int>(seqlen_count), gsl::narrow_cast<int>(head_count));
  }

  ORT_THROW("SoftmaxVarLengthComputeHelper: Unsupported input shape");
}

#define SPECIALIZED_SOFTMAX_HELPER_IMPL(T, TOut)                                             \
  template Status SoftMaxVarLengthComputeHelper<T, TOut, false>(Stream * stream,             \
                                                                const T* X,                  \
                                                                const int64_t max_seq_lengh, \
                                                                const int64_t* cu_seqlens,   \
                                                                const int64_t seqlen_count,  \
                                                                const int64_t head_count,    \
                                                                TOut* Y);

SPECIALIZED_SOFTMAX_HELPER_IMPL(MLFloat16, float)
SPECIALIZED_SOFTMAX_HELPER_IMPL(float, float)
SPECIALIZED_SOFTMAX_HELPER_IMPL(double, double)
SPECIALIZED_SOFTMAX_HELPER_IMPL(MLFloat16, MLFloat16)
SPECIALIZED_SOFTMAX_HELPER_IMPL(BFloat16, BFloat16)

}  // namespace cuda
}  // namespace onnxruntime
