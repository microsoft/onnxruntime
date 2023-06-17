// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/quantization/fake_quant_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

namespace {
constexpr int NumElementsPerThread = GridDim::maxElementsPerThread;
constexpr int NumThreadsPerBlock = GridDim::maxThreadsPerBlock;
}  // namespace

template <typename T>
__global__ void FakeQuantPerTensorImpl(const int64_t num_elements, const T* input_data, const T quant_scale,
                                       const T quant_zero_point, const int64_t quant_min, const int64_t quant_max,
                                       T* fake_quantized_data, bool* quantization_mask_data) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  T values[NumElementsPerThread];
  T fake_quantized_values[NumElementsPerThread];
  bool mask_values[NumElementsPerThread];

  CUDA_LONG idx = start;
  // Load
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (idx < num_elements) {
      values[i] = input_data[idx];
      idx += NumThreadsPerBlock;
    }
  }

  // Compute
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    // Quantize
    const auto quantized_value = std::nearbyint(values[i] / quant_scale) + quant_zero_point;
    // Clamp and De-Quantize
    fake_quantized_values[i] =
        (fminf(quant_max, fmaxf(quant_min, quantized_value)) - quant_zero_point) * quant_scale;
    // Compute mask
    mask_values[i] = (quant_min <= quantized_value && quantized_value <= quant_max);
  }

  // Write
  idx = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (idx < num_elements) {
      fake_quantized_data[idx] = fake_quantized_values[i];
      quantization_mask_data[idx] = mask_values[i];
      idx += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void FakeQuantPerTensor(cudaStream_t stream, const int64_t num_elements, const T* input_data, const T quant_scale,
                        const T quant_zero_point, const int64_t quant_min, const int64_t quant_max,
                        T* fake_quantized_data, bool* quantization_mask_data) {
  int blocksPerGrid =
      static_cast<int>(CeilDiv(num_elements, NumThreadsPerBlock * NumElementsPerThread));
  FakeQuantPerTensorImpl<T><<<blocksPerGrid, NumThreadsPerBlock, 0, stream>>>(
      num_elements, input_data, quant_scale, quant_zero_point,
      quant_min, quant_max, fake_quantized_data, quantization_mask_data);
}

#define SPECIALIZED_FAKEQUANT_IMPL(T)                                                    \
  template void FakeQuantPerTensor<T>(cudaStream_t stream, const int64_t num_elements,   \
                                      const T* input_data, const T quant_scale,          \
                                      const T quant_zero_point, const int64_t quant_min, \
                                      const int64_t quant_max, T* fake_quantized_data,   \
                                      bool* quantization_mask_data);

SPECIALIZED_FAKEQUANT_IMPL(float)

#undef SPECIALIZED_FAKEQUANT_IMPL

template <typename T>
struct FakeQuantGradFunctor {
  FakeQuantGradFunctor(const T* dY_data, const bool* gradient_mask_data)
      : dY_data_(dY_data),
        gradient_mask_data_(gradient_mask_data) {}

  __device__ __inline__ T operator()(CUDA_LONG idx) const {
    // If gradient_mask is true (i.e. quantization was in range), return dY, else return 0
    return gradient_mask_data_[idx] ? dY_data_[idx] : static_cast<T>(0);
  }

  const T* dY_data_;
  const bool* gradient_mask_data_;
};

template <typename T>
void FakeQuantGradImpl(cudaStream_t stream, const int64_t num_elements, const T* dY_data,
                       const bool* gradient_mask_data, T* dX_data) {
  FakeQuantGradFunctor<T> fake_quant_grad_functor(dY_data, gradient_mask_data);
  LaunchElementwiseKernel<T, decltype(fake_quant_grad_functor), CUDA_LONG>(
      stream, dX_data, fake_quant_grad_functor, static_cast<CUDA_LONG>(num_elements));
}

#define SPECIALIZED_FAKEQUANTGRAD_IMPL(T)                                             \
  template void FakeQuantGradImpl<T>(cudaStream_t stream, const int64_t num_elements, \
                                     const T* dY_data, const bool* gradient_mask_data, T* dX_data);

SPECIALIZED_FAKEQUANTGRAD_IMPL(float)

#undef SPECIALIZED_FAKEQUANTGRAD_IMPL

}  // namespace cuda
}  // namespace onnxruntime
