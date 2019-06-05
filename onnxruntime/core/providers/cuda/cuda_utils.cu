// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Thrust code needs to be compiled with nvcc
#include <memory>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _Fill(
    T* output_data,
    T val,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = val;
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

  virtual const T* GetBuffer(size_t count) {
    if (count > count_) {
      if (buffer_) {
        cudaFree(buffer_);
        buffer_ = nullptr;
      }
      CUDA_CALL_THROW(cudaMalloc(&buffer_, count * sizeof(T)));
      count_ = count;

      int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
      CUDA_LONG N = static_cast<CUDA_LONG>(count);
      _Fill<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(buffer_, val_, N);
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
  return std::make_unique<ConstantBufferImpl<T>>(Consts<T>::One);
}

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantValue(T value) {
  return std::make_unique<ConstantBufferImpl<T>>(value);
}

template std::unique_ptr<IConstantBuffer<float>> CreateConstantOnes<float>();
template std::unique_ptr<IConstantBuffer<double>> CreateConstantOnes<double>();
template std::unique_ptr<IConstantBuffer<half>> CreateConstantOnes<half>();

template std::unique_ptr<IConstantBuffer<float>> CreateConstantValue<float>(float value);
template std::unique_ptr<IConstantBuffer<double>> CreateConstantValue<double>(double value);
template std::unique_ptr<IConstantBuffer<half>> CreateConstantValue<half>(half value);
template std::unique_ptr<IConstantBuffer<bool>> CreateConstantValue<bool>(bool value);
template std::unique_ptr<IConstantBuffer<int8_t>> CreateConstantValue<int8_t>(int8_t value);
template std::unique_ptr<IConstantBuffer<int16_t>> CreateConstantValue<int16_t>(int16_t value);
template std::unique_ptr<IConstantBuffer<int32_t>> CreateConstantValue<int32_t>(int32_t value);
template std::unique_ptr<IConstantBuffer<int64_t>> CreateConstantValue<int64_t>(int64_t value);
template std::unique_ptr<IConstantBuffer<uint8_t>> CreateConstantValue<uint8_t>(uint8_t value);
template std::unique_ptr<IConstantBuffer<uint16_t>> CreateConstantValue<uint16_t>(uint16_t value);
template std::unique_ptr<IConstantBuffer<uint32_t>> CreateConstantValue<uint32_t>(uint32_t value);
template std::unique_ptr<IConstantBuffer<uint64_t>> CreateConstantValue<uint64_t>(uint64_t value);

}  // namespace cuda
}  // namespace onnxruntime
