// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/quantization/fake_quant_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

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

}  // namespace cuda
}  // namespace onnxruntime
