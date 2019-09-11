#include "softmax_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename T, int blockSize, int N>
__launch_bounds__(blockSize)
    __global__ void MHASoftmaxKernel(const T* input, T* output) {
  int stride = N;
  const int NperThr = N / blockSize;
  const T* input_start = input + stride * blockIdx.x;
  T* output_start = output + stride * blockIdx.x;
  int tid = threadIdx.x;
  float logits_tmp[NperThr];
  __shared__ float sum_shared[blockSize];
  sum_shared[tid] = 0.;
  int idx_start = tid * NperThr;
  for (int i = idx_start; i < idx_start + NperThr; i++) {
    float x = input_start[i];
    logits_tmp[i - idx_start] = expf(x);
    sum_shared[tid] += logits_tmp[i - idx_start];
  }
  __syncthreads();
  if (tid == 0) {
    for (int i = 1; i < blockSize; ++i) {
      sum_shared[0] += sum_shared[i];
    }
  }
  __syncthreads();
  float recip_sum = 1.f / sum_shared[0];
  for (int i = idx_start; i < idx_start + NperThr; i++) {
    output_start[i] = logits_tmp[i - idx_start] * recip_sum;
  }
}

template <typename T, int blockSize>
__launch_bounds__(blockSize)
    __global__ void MHASoftmaxKernelGeneric(const T* input, T* output, int dim1) {
  int N = dim1;
  int stride = N;
  const T* input_start = input + stride * blockIdx.x;
  T* output_start = output + stride * blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float sum_shared[blockSize];
  sum_shared[tid] = 0.;
  for (int i = tid; i < N; i += blockSize) {
    float x = input_start[i];
    x = expf(x);
    sum_shared[tid] += x;
    output_start[i] = (T)x;
  }
  __syncthreads();
  if (tid == 0) {
    for (int i = 1; i < blockSize; ++i) {
      sum_shared[0] += sum_shared[i];
    }
  }
  __syncthreads();
  float recip_sum = 1.f / sum_shared[0];
  for (int i = tid; i < N; i += blockSize) {
    output_start[i] = (T)(recip_sum * (float)output_start[i]);
  }
}

template <class T>
void launchSoftmaxKernel(const T* input,
                         T* output,
                         int N, int D) {
  const int blockSize = 32;
  const int gridSize = N;

  // Specialized kernel for sequence length = 128.
  // Add more sequence lengths as necessary for better softmax performance.
  if (D == 128) {
    MHASoftmaxKernel<T, blockSize, 128><<<gridSize, blockSize, 0>>>(input, output);
  } else if (D == 32) {
    MHASoftmaxKernel<T, blockSize, 32><<<gridSize, blockSize, 0>>>(input, output);
  } else {
    MHASoftmaxKernelGeneric<T, blockSize><<<gridSize, blockSize, 0>>>(input, output, D);
  }
}

template void launchSoftmaxKernel( const float* input, float* output, int N, int D );
template void launchSoftmaxKernel( const double* input, double* output, int N, int D );
template void launchSoftmaxKernel( const half* input, half* output, int N, int D );

}  // namespace cuda
}  // namespace onnxruntime
