// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
#ifdef USE_ROCM
constexpr int kThreadsPerBlock = 512;
#else
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
#endif

}  // namespace

// Need to use SplitSameSplitDimImpl (the other one works for different split sizes)

template <typename scalar_t>
__global__ void S2SModelSplitQuickGeluKernel(scalar_t* output, scalar_t* input, uint dim) {
  uint input_line_stride = dim * 2;
  uint output_line_stride = dim;
  uint offset_in1 = blockIdx.x * input_line_stride + threadIdx.x*kElementsPerThread;
  uint offset_in2 = offset_in1 + dim;
  uint offset_out = blockIdx.x * output_line_stride + threadIdx.x*kElementsPerThread;
  for (uint i = 0; i < kElementsPerThread; i++) {
    if (offset_out < dim) {
      scalar_t v = input[offset_in2 + i] * static_cast<scalar_t>(alpha);
      scalar_t one = static_cast<scalar_t>(1.f);
      scalar_t zero = static_cast<scalar_t>(0.f);
      scalar_t sigmoid = v >= zero ? one / (one + _Exp(-v)) : one - one / (one + _Exp(v));
      output[offset_out + i] = input[offset_in1 + i] * sigmoid;
    }
  }
}


template <typename T, typename OutputDataArray>
__global__ void S2SModelSplitQuickGeluKernel_old(const fast_divmod block_size_including_axis_dim_div,
                                            const fast_divmod block_size_inside_axis_dim_div,
                                            const fast_divmod split_dim_size, const int num_outputs, const T* input_data,
                                            OutputDataArray output_data, const CUDA_LONG N) {
  CUDA_LONG start = kNumElementsPerThread * kNumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kNumElementsPerThread];

  // T s1[kNumElementsPerThread];
  // T s2[kNumElementsPerThread];

  // s1, s2 = T* output_data
  // T s1 =
  // uint offset_s1, o_s2;
  // s1 = input_data[o_s1]





  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      value[i] = input_data[id];
      id += kNumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N) {
      int outer_block_index, block_index, offset, output_index, block_offset;
      block_size_including_axis_dim_div.divmod(id, outer_block_index, offset);
      block_size_inside_axis_dim_div.divmod(offset, block_index, offset);
      split_dim_size.divmod(block_index, output_index, block_offset);
      CUDA_LONG output_pos =
          (outer_block_index * split_dim_size.d_ + block_offset) * block_size_inside_axis_dim_div.d_ + offset;
      // reinterpret_cast<T*>(output_data[output_index])[output_pos] = value[i];
      // if output_index == 0:
      //   s1[i] = input_data[id];
      // else:
      //  s2[i] = Quickgelu(input_data[id]);
      //
      id += kNumThreadsPerBlock;
    }
  }

  // Should call cuda sync or something similar to ensure all threads have completed till here
  __syncthreads();

// #pragma unroll
  // output_data[id] = s1[i] * s2[i]

  // Here the output should have two indices, the first one should go directly to Multiply
  // Second one should go QuickGelu operator

//   id = start;
// CUDA_LONG id = start;
// #pragma unroll
//   for (int i = 0; i < kNumElementsPerThread; ++i) {
//     if (id < N) {
//       value[i] = output_data[num_outputs-1][id];
//       id += kNumThreadsPerBlock;
//     }
//   }


#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N/num_outputs) {
      // Should it be id or output_pos?
      // Should I be using reinterpret_cast?
      T v = output_data[num_outputs-1][id] * static_cast<T>(alpha);
      T one = static_cast<T>(1.f);
      T zero = static_cast<T>(0.f);
      T sigmoid = v >= zero ? one / (one + _Exp(-v)) : one - one / (one + _Exp(v));
      // Should I overwrite?
      output_data[num_outputs-1][id] = v * sigmoid;
    }
    id += kNumThreadsPerBlock;
    // Is this the right way to do it?
    if (id > N/2) {
      break;
    }
  };

  __syncthreads();

  // Now multiply
  // Final output_data_final
  id = start;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    if (id < N/2) {
      output_data_final[id] = output_data[0][id] * output_data[1][id]
    }
    id += kNumThreadsPerBlock;
    // Is this the right way to do it?
    if (id > N/2) {
      break;
    }
  };


  // Need to be removed code below
  int idx = ??;
  if (idx < input_length) {
    const T x = output_data[1][idx];
    const T in = (bias == nullptr) ? x : (T)(x + bias[idx % bias_length]);
    const T cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }

  OutputDataArray v = output_data[1] * static_cast<T>(alpha);
  OutputDataArray one = static_cast<OutputDataArray>(1.f);
  OutputDataArray zero = static_cast<OutputDataArray>(0.f);
  OutputDataArray sigmoid = v >= zero ? one / (one + _Exp(-v)) : one - one / (one + _Exp(v));
  OutputDataArray quick_gelu_out = output_data[1] * sigmoid;





  const auto kElementsPerBlock = kElementsPerThread * blockDim.x;
  // const auto input_base_idx = bias_size * blockIdx.x + kElementsPerBlock * blockIdx.y + threadIdx.x;
  // const auto bias_base_idx = kElementsPerBlock * blockIdx.y + threadIdx.x;
  const auto element_stride = blockDim.x;

  T reg_X[kElementsPerThread];

  {
    // auto input_idx = input_base_idx;
    // auto bias_idx = bias_base_idx;
#pragma unroll
    for (int element_idx = 0; element_idx < kElementsPerThread; ++element_idx) {
      if (bias_idx < bias_size) {
        reg_X[element_idx] = X[input_idx];
        reg_B[element_idx] = B[bias_idx];
        input_idx += element_stride;
        bias_idx += element_stride;
      }
    }
  }

  {
    auto input_idx = input_base_idx;
    auto bias_idx = bias_base_idx;
#pragma unroll
    for (int element_idx = 0; element_idx < kElementsPerThread; ++element_idx) {
      if (bias_idx < bias_size) {
        Y[input_idx] = _Gelu(reg_X[element_idx] + reg_B[element_idx]);
        input_idx += element_stride;
        bias_idx += element_stride;
      }
    }
  }
}

// template <typename T>
// __global__ void VectorizedS2SModelSplitQuickGeluKernel(int64_t axis, const T* X, T* Y) {
//   const auto kElementsPerBlock = kElementsPerThread * blockDim.x;
//   const auto bias_idx = kElementsPerBlock * blockIdx.y + kElementsPerThread * threadIdx.x;
//   if (bias_idx >= bias_size) {
//     return;
//   }

//   const auto input_idx = bias_size * blockIdx.x + kElementsPerBlock * blockIdx.y + kElementsPerThread * threadIdx.x;

//   using LoadT = aligned_vector<T, kElementsPerThread>;

//   T reg_X[kElementsPerThread];
//   T reg_Y[kElementsPerThread];

//   LoadT* value_X = reinterpret_cast<LoadT*>(&reg_X);
//   LoadT* value_B = reinterpret_cast<LoadT*>(&reg_B);
//   *value_X = *reinterpret_cast<const LoadT*>(&X[input_idx]);
//   *value_B = *reinterpret_cast<const LoadT*>(&B[bias_idx]);

// #pragma unroll
//   for (int element_idx = 0; element_idx < kElementsPerThread; ++element_idx) {
//     reg_Y[element_idx] = _Gelu(reg_X[element_idx] + reg_B[element_idx]);
//   }

//   *(reinterpret_cast<LoadT*>(&Y[input_idx])) = *reinterpret_cast<LoadT*>(&reg_Y[0]);
// }

// void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream, const size_t element_size, const int block_size_including_axis_dim,
//                                         const int block_size_inside_axis_dim, const int64_t split_size, const int num_outputs,
//                                         const void* input_data, OutputDataArray output_data, const size_t input_size) {

template <typename T>
void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream,
                                        const int num_outputs,
                                        const void* input_data, void* output_data) {
  CUDA_LONG N = static_cast<CUDA_LONG>(input_size);
  int blocksPerGrid = CeilDiv(N, kNumElementsPerThread * kNumThreadsPerBlock);
  fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);
  fast_divmod split_size_div = fast_divmod(static_cast<int>(split_size));

  switch (element_size) {
#define CASE_ELEMENT_TYPE(type)                                                                         \
  case sizeof(type): {                                                                                  \
    S2SModelSplitQuickGeluKernel<<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(                    \
        block_size_including_axis_dim_div, block_size_inside_axis_dim_div, split_size_div, num_outputs, \
        reinterpret_cast<const ToCudaType<type>::MappedType*>(input_data), output_data, N);             \
  } break
    CASE_ELEMENT_TYPE(int8_t);
    CASE_ELEMENT_TYPE(int16_t);
    CASE_ELEMENT_TYPE(int32_t);
    CASE_ELEMENT_TYPE(int64_t);
#undef CASE_ELEMENT_TYPE
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Split QuickGelu Fusion operator");
  }

  return Status::OK();



  // TODO: Call Split Function, it will have two outputs, out1, out2

  // TODO: Call QuickGelu on second output (OP_QuickGelu/CtxQuickGelu)
  // store it as out_quickgelu

  // TODO: Multiply out1 and out_quickgelu and store it in output Y

  T reg_Y[kElementsPerThread] = out1 * out_quickgelu;
  *(reinterpret_cast<LoadT*>(&Y[input_idx])) = *reinterpret_cast<LoadT*>(&reg_Y[0]);


  int num_threads_per_block = std::min<int>(static_cast<int>(CeilDiv(bias_size, kElementsPerThread)), kThreadsPerBlock);
  const auto grid_width = CeilDiv(bias_size, kElementsPerThread * num_threads_per_block);
  const auto grid_height = input_size / bias_size;
  const dim3 grid_dim{static_cast<uint32_t>(grid_height), static_cast<uint32_t>(grid_width)};

  constexpr int vec_alignment = std::alignment_of<aligned_vector<T, kElementsPerThread>>::value;

  // Calling the Split kernel
  S2SModelSplitQuickGeluKernel<T><<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(
    block_size_including_axis_dim_div, block_size_inside_axis_dim_div, split_size_div, num_outputs,
    reinterpret_cast<const ToCudaType<type>::MappedType*>(input_data), output_data, N
  )


  if (bias_size % kElementsPerThread == 0 && reinterpret_cast<uint64_t>(X) % vec_alignment == 0 &&
      reinterpret_cast<uint64_t>(Y) % vec_alignment == 0) {
    VectorizedS2SModelSplitQuickGeluKernel<T><<<grid_dim, num_threads_per_block, 0, stream>>>(bias_size, X, Y);
  } else {
    S2SModelSplitQuickGeluKernel<T><<<grid_dim, num_threads_per_block, 0, stream>>>(bias_size, X, Y);
  }
}

// explicit instantiations
#define SPECIALIZED_SPLIT_QUICKGELU_IMPL(T)                                                                           \
  template void LaunchS2SModelSplitQuickGeluKernel<T>(cudaStream_t stream, int64_t input_size, int64_t axis,          \
                                                      int64_t alpha, const T* X, const T* S, T* Y)

SPECIALIZED_SPLIT_QUICKGELU_IMPL(half);
SPECIALIZED_SPLIT_QUICKGELU_IMPL(float);
SPECIALIZED_SPLIT_QUICKGELU_IMPL(double);
SPECIALIZED_SPLIT_QUICKGELU_IMPL(BFloat16);

#undef SPECIALIZED_SPLIT_QUICKGELU_IMPL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
