// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
// #ifdef USE_ROCM
// constexpr int kThreadsPerBlock = 512;
// #else
// constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
// #endif

}  // namespace

// Need to use SplitSameSplitDimImpl (the other one works for different split sizes)

template <typename T>
__global__ void S2SModelSplitQuickGeluKernel(const int num_outputs, const T* input, T* output) {
  uint dim = 2;
  uint max_len = 16;
  float alpha = 1.702f;
  uint max_dim = 4;
  T one = static_cast<T>(1.f);
  T zero = static_cast<T>(0.f);
  for (uint i = 0; i < max_dim; i++){
    for (uint j = 0; j < dim; j++){
      T v = input[dim + i*max_dim+j] * static_cast<T>(alpha);
      T sigmoid = v >= zero ? one / (one + _Exp(-v)) : one - one / (one + _Exp(v));
      T quickgelu_out = input[dim + i*max_dim+j] * sigmoid;
      output[i*max_dim/2+j] = input[i*max_dim+j] * quickgelu_out;
    }
  }
  // for (uint i = 0; i < max_len / 2; i++) {
  //   T v = input[dim + i] * static_cast<T>(alpha);
  //   T one = static_cast<T>(1.f);
  //   T zero = static_cast<T>(0.f);
  //   T sigmoid = v >= zero ? one / (one + _Exp(-v)) : one - one / (one + _Exp(v));
  //   output[i] = input[i] * sigmoid;
  // }
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

// void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream, const size_t element_size, const int block_size_including_axis_dim,
//                                         const int block_size_inside_axis_dim, const int64_t split_size, const int num_outputs,
//                                         const void* input_data, OutputDataArray output_data, const size_t input_size) {

template <typename T>
void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream, int num_outputs, const T* input_data, T* output_data) {
  // CUDA_LONG N = static_cast<CUDA_LONG>(input_size);
  // int blocksPerGrid = CeilDiv(N, kNumElementsPerThread * kNumThreadsPerBlock);
  // fast_divmod block_size_including_axis_dim_div = fast_divmod(block_size_including_axis_dim);
  // fast_divmod block_size_inside_axis_dim_div = fast_divmod(block_size_inside_axis_dim);
  // fast_divmod split_size_div = fast_divmod(static_cast<int>(split_size));
  S2SModelSplitQuickGeluKernel<T><<<1, 1, 0, stream>>>(num_outputs, input_data, output_data);



  // return Status::OK();



  // TODO: Call Split Function, it will have two outputs, out1, out2

  // TODO: Call QuickGelu on second output (OP_QuickGelu/CtxQuickGelu)
  // store it as out_quickgelu

  // TODO: Multiply out1 and out_quickgelu and store it in output Y

  // T reg_Y[kElementsPerThread] = out1 * out_quickgelu;
  // *(reinterpret_cast<LoadT*>(&Y[input_idx])) = *reinterpret_cast<LoadT*>(&reg_Y[0]);


  // int num_threads_per_block = std::min<int>(static_cast<int>(CeilDiv(bias_size, kElementsPerThread)), kThreadsPerBlock);
  // const auto grid_width = CeilDiv(bias_size, kElementsPerThread * num_threads_per_block);
  // const auto grid_height = input_size / bias_size;
  // const dim3 grid_dim{static_cast<uint32_t>(grid_height), static_cast<uint32_t>(grid_width)};

  // constexpr int vec_alignment = std::alignment_of<aligned_vector<T, kElementsPerThread>>::value;

  // // Calling the Split kernel
  // S2SModelSplitQuickGeluKernel<T><<<blocksPerGrid, kNumThreadsPerBlock, 0, stream>>>(
  //   block_size_including_axis_dim_div, block_size_inside_axis_dim_div, split_size_div, num_outputs,
  //   reinterpret_cast<const ToCudaType<type>::MappedType*>(input_data), output_data, N
  // )
}

// explicit instantiations
#define SPECIALIZED_SplitQuickGelu_IMPL(T)                                                   \
  template void LaunchS2SModelSplitQuickGeluKernel<T>(cudaStream_t stream, int num_outputs,  \
                                                      const T* input_data, T* output_data)

SPECIALIZED_SplitQuickGelu_IMPL(float);
SPECIALIZED_SplitQuickGelu_IMPL(half);
SPECIALIZED_SplitQuickGelu_IMPL(BFloat16);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
