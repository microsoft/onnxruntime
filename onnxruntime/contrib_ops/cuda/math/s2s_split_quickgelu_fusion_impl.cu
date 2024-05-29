// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
// #ifdef USE_ROCM
// constexpr int kThreadsPerBlock = 512;
// #else
// constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
// #endif

}  // namespace

// Need to use SplitSameSplitDimImpl (the other one works for different split sizes)

template <typename T>
__global__ void S2SModelSplitQuickGeluKernel(const int dim, const T* input, T* output) {
  // Can remove num_outputs parameter
  // CHange dim to be part of input
  // uint dim = 2;
  // printf("Output dim is %d\n", dim);
  uint input_line_stride = dim * 2;
  uint output_line_stride = dim;
  uint offset_in1 = blockIdx.x * input_line_stride + threadIdx.x*kElementsPerThread;
  // 10 el (dim = 5)
  // 0 idx, 5 idx
  //
  uint offset_in2 = offset_in1 + dim;
  uint offset_out = blockIdx.x * output_line_stride + threadIdx.x*kElementsPerThread;
  T one = static_cast<T>(1.f);
  T zero = static_cast<T>(0.f);
  // Specify alpha here or outside (is this an input )
  float alpha = 1.702f;
  T alpha_val = static_cast<T>(alpha);
  // Separate QuickGelu code in another fn
  printf("Curr kElementsPerThread %d\n", kElementsPerThread);
  printf("Curr blockIdx.x %d\n", blockIdx.x);
  printf("Curr threadIdx.x %d\n", threadIdx.x);
  printf("Curr offset_in1 %d\n", offset_in1);
  printf("Curr offset_in2 %d\n", offset_in2);
  printf("Curr offset_out %d\n", offset_out);
  // std::cout << "Curr kElementsPerThread:" << kElementsPerThread << std::endl;
  // input_size - dim
  // 5x4 (input_size = 20), dim = 2
  // 20 - 2
  // output is 5x 10 (dim = 10)
  // 5 is row number
  // 1 cuda block will process 1 row
  // kElementsPerThread
  // dim = 10K
  //
  // if threadIdx.x*kElementsPerThread < dim
  // int max_inp = 20 - dim;
  // What about this condition? (Removing if condition should improve Warp Divergence?)
  // for (uint i = 0; i < kElementsPerThread; i++) {
  for (uint i = 0; i < kElementsPerThread && threadIdx.x*kElementsPerThread + i < dim; i++){
    uint curr_in = offset_in1 + i;
    // int curr_half = curr_in / dim;
    printf("Curr Inp Outside %d\n", curr_in);
    // if (curr_half %2 == 0 && curr_in < max_inp){
    if (threadIdx.x*kElementsPerThread + i < dim) {
      printf("Curr Inp inside %d\n", curr_in);
      // std::cout << "Curr curr_in:" << curr_in << std::endl;
      T v = input[offset_in2+i] * alpha_val;
      T sigmoid = v >= zero ? one / (one + _Exp(-v)) : one - one / (one + _Exp(v));
      T quickgelu_out = input[offset_in2+i] * sigmoid;
      output[offset_out + i] = input[offset_in1 + i] * quickgelu_out;
      printf("Current output idx %d\n", offset_out + i);
      // printf("Current output value %f\n", quickgelu_out);
    }
  }

  // uint dim = 2;
  // float alpha = 1.702f;
  // uint twice_dim = 2*dim;
  // // max_dim is input size divided by 2*dim
  // uint max_dim = 5;
  // T one = static_cast<T>(1.f);
  // T zero = static_cast<T>(0.f);
  // for (uint i = 0; i < max_dim; i++){
  //   for (uint j = 0; j < dim; j++){
  //     T v = input[dim + i*twice_dim+j] * static_cast<T>(alpha);
  //     T sigmoid = v >= zero ? one / (one + _Exp(-v)) : one - one / (one + _Exp(v));
  //     T quickgelu_out = input[dim + i*twice_dim+j] * sigmoid;
  //     output[i*dim+j] = input[i*twice_dim+j] * quickgelu_out;
  //   }
  // }
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
void LaunchS2SModelSplitQuickGeluKernel(cudaStream_t stream, int dim, int64_t input_size, const T* input_data, T* output_data) {
  CUDA_LONG N = static_cast<CUDA_LONG>(input_size);
  int num_threads_per_block = std::min<int>(static_cast<int>(CeilDiv(dim, kElementsPerThread)), kThreadsPerBlock);
  int num_blocks = static_cast<int>(N/(2*dim));
  printf("Final number threads per block %d\n", num_threads_per_block);
  printf("Final num blocks %d\n", num_blocks);
  S2SModelSplitQuickGeluKernel<T><<<num_blocks, num_threads_per_block, 0, stream>>>(dim, input_data, output_data);
  // S2SModelSplitQuickGeluKernel<T><<<5, 1, 0, stream>>>(dim, input_data, output_data);
  // 4x10
  // output_dim = 5
  // 1st number of blocks = 4 (number of rows = 4)
  // 2nd size of block (threadsize) = ceil(5 / kElementsPerThread)
  // 3rd is don't need dynamic size
  //
  // 4x5x10
  // 20x10
  // number of blocks = 20
  // threadsize = ceil(5/4) = 2




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
  template void LaunchS2SModelSplitQuickGeluKernel<T>(cudaStream_t stream, int dim, int64_t input_size, \
                                                      const T* input_data, T* output_data)

SPECIALIZED_SplitQuickGelu_IMPL(float);
SPECIALIZED_SplitQuickGelu_IMPL(half);
SPECIALIZED_SplitQuickGelu_IMPL(BFloat16);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
