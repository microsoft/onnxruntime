// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Thrust code needs to be compiled with nvcc
#include <memory>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _Fill(
    T* output_data,
    T val,
    CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = val;
      id += blockDim.x;
    }
  }
}

template <typename T>
void Fill(cudaStream_t stream, T* output, T value, int64_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _Fill<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(output, value, N);
}
template <typename T>
class ConstantBufferImpl : public IConstantBuffer<T> {
 public:
  ConstantBufferImpl(T val) : val_(val), buffer_(nullptr), count_(0) {
  }
  ~ConstantBufferImpl() {
    if (buffer_)
      cudaFree(buffer_);
  }

  virtual const T* GetBuffer(cudaStream_t stream, size_t count) {
    if (count > count_) {
      if (buffer_) {
        cudaFree(buffer_);
        buffer_ = nullptr;
      }
      CUDA_CALL_THROW(cudaMalloc(&buffer_, count * sizeof(T)));
      count_ = count;

      Fill(stream, buffer_, val_, count);
    }
    return buffer_;
  }

 private:
  T* buffer_;
  size_t count_;
  T val_;
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantOnes() {
  return onnxruntime::make_unique<ConstantBufferImpl<T>>(Consts<T>::One);
}

template std::unique_ptr<IConstantBuffer<float>> CreateConstantOnes<float>();
template std::unique_ptr<IConstantBuffer<double>> CreateConstantOnes<double>();
template std::unique_ptr<IConstantBuffer<half>> CreateConstantOnes<half>();
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template std::unique_ptr<IConstantBuffer<nv_bfloat16>> CreateConstantOnes<nv_bfloat16>();
#endif

#define SPECIALIZED_FILL(T) \
  template void Fill<T>(cudaStream_t stream, T * output, T value, int64_t count);

SPECIALIZED_FILL(int8_t)
SPECIALIZED_FILL(int16_t)
SPECIALIZED_FILL(int32_t)
SPECIALIZED_FILL(int64_t)
SPECIALIZED_FILL(float)
SPECIALIZED_FILL(double)
SPECIALIZED_FILL(__half)
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
SPECIALIZED_FILL(nv_bfloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime
