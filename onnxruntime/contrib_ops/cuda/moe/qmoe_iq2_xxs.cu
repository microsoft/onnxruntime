// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/qmoe_iq2_xxs.h"
#include "contrib_ops/cuda/quantization/iq2_xxs_decode.cuh"
#include "core/providers/cuda/cuda_common.h"

#include <algorithm>

namespace onnxruntime::contrib::cuda {
namespace {

using namespace onnxruntime::cuda;

__global__ void ValidateRoutes(const int64_t* ids, const float* weights,
                               int64_t tokens, int64_t experts, int k, int* error) {
  for (int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < tokens; token += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int invalid = 0;
    for (int r = 0; r < k; ++r) {
      const int64_t id = ids[token * k + r];
      if (id < 0 || id >= experts) {
        invalid |= 1;
      }
      if (!isfinite(weights[token * k + r])) {
        invalid |= 2;
      }
      for (int previous = 0; previous < r; ++previous) {
        if (id == ids[token * k + previous]) {
          invalid |= 4;
        }
      }
    }
    if (invalid != 0) {
      atomicOr(error, invalid);
    }
  }
}

__global__ void SortRoutes(const int64_t* ids, int* order, int64_t tokens, int k) {
  for (int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       token < tokens; token += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int sorted[kIQ2XXSMaxSelectedExperts];
    for (int r = 0; r < k; ++r) {
      int position = r;
      while (position > 0 && ids[token * k + sorted[position - 1]] > ids[token * k + r]) {
        sorted[position] = sorted[position - 1];
        --position;
      }
      sorted[position] = r;
    }
    for (int r = 0; r < k; ++r) {
      order[token * k + r] = sorted[r];
    }
  }
}

template <typename T>
__global__ void SelectedUpRelu2(const T* input, const uint8_t* bank,
                                const int64_t* ids, T* intermediate,
                                int64_t tokens, int64_t hidden, int64_t inter, int k) {
  const int lane = threadIdx.x % 32;
  const int64_t k_blocks = hidden / 256;
  for (int64_t output = static_cast<int64_t>(blockIdx.x) * 8 + threadIdx.x / 32;
       output < tokens * k * inter; output += static_cast<int64_t>(gridDim.x) * 8) {
    const int64_t route = output / inter;
    const int64_t expert = ids[route];
    const int64_t col = output % inter;
    const float sum = iq2_xxs::Dot(
        input + (route / k) * hidden, bank + (expert * inter + col) * k_blocks * 66, k_blocks);
    if (lane == 0) {
      const float rounded = to_float<T>(from_float<T>(sum));
      const float relu = rounded < 0.0f ? 0.0f : rounded;
      intermediate[output] = from_float<T>(relu * relu);
    }
  }
}

template <typename T>
__global__ void SelectedDownReduce(const T* intermediate, const uint8_t* bank,
                                   const int64_t* ids, const float* weights,
                                   const int* order, T* output,
                                   int64_t tokens, int64_t hidden, int64_t inter, int k) {
  const int lane = threadIdx.x % 32;
  const int64_t k_blocks = inter / 256;
  for (int64_t out = static_cast<int64_t>(blockIdx.x) * 8 + threadIdx.x / 32;
       out < tokens * hidden; out += static_cast<int64_t>(gridDim.x) * 8) {
    const int64_t token = out / hidden;
    const int64_t col = out % hidden;
    float total = 0.0f;
    for (int r = 0; r < k; ++r) {
      const int64_t route = token * k + order[token * k + r];
      const int64_t expert = ids[route];
      const float sum = iq2_xxs::Dot(
          intermediate + route * inter, bank + (expert * hidden + col) * k_blocks * 66, k_blocks);
      if (lane == 0) {
        const float rounded = to_float<T>(from_float<T>(sum));
        // Preserve separate FP32 Mul/Add nodes rather than a contracted FMA.
        total = __fadd_rn(total, __fmul_rn(rounded, weights[route]));
      }
    }
    if (lane == 0) {
      output[out] = from_float<T>(total);
    }
  }
}

int Blocks(int64_t work, int per_block) {
  return static_cast<int>(std::min<int64_t>((work - 1) / per_block + 1, 65535));
}

template <typename T>
Status Launch(const void* input, const uint8_t* up, const uint8_t* down,
              const int64_t* ids, const float* weights, void* output,
              void* intermediate, int* order, int64_t tokens,
              int64_t hidden, int64_t inter, int k, cudaStream_t stream) {
  SortRoutes<<<Blocks(tokens, 128), 128, 0, stream>>>(ids, order, tokens, k);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  SelectedUpRelu2<<<Blocks(tokens * k * inter, 8), 256, 0, stream>>>(
      static_cast<const T*>(input), up, ids, static_cast<T*>(intermediate), tokens, hidden, inter, k);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  SelectedDownReduce<<<Blocks(tokens * hidden, 8), 256, 0, stream>>>(
      static_cast<const T*>(intermediate), down, ids, weights, order, static_cast<T*>(output),
      tokens, hidden, inter, k);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace

Status ValidateQMoEIQ2XXSRoutes(const int64_t* ids, const float* weights,
                                int64_t tokens, int64_t experts, int k,
                                int* error, cudaStream_t stream) {
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(error, 0, sizeof(int), stream));
  ValidateRoutes<<<Blocks(tokens, 128), 128, 0, stream>>>(ids, weights, tokens, experts, k, error);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

Status LaunchQMoEIQ2XXS(const void* input, const uint8_t* up, const uint8_t* down,
                        const int64_t* ids, const float* weights, void* output,
                        void* intermediate, int* order, int64_t tokens,
                        int64_t hidden, int64_t inter, int k,
                        IQ2XXSActivationType type, cudaStream_t stream) {
  switch (type) {
    case IQ2XXSActivationType::Float:
      ORT_RETURN_IF_ERROR(Launch<float>(input, up, down, ids, weights, output, intermediate, order,
                                        tokens, hidden, inter, k, stream));
      break;
    case IQ2XXSActivationType::Float16:
      ORT_RETURN_IF_ERROR(Launch<half>(input, up, down, ids, weights, output, intermediate, order,
                                       tokens, hidden, inter, k, stream));
      break;
    case IQ2XXSActivationType::BFloat16:
      ORT_RETURN_IF_ERROR(Launch<nv_bfloat16>(input, up, down, ids, weights, output, intermediate, order,
                                              tokens, hidden, inter, k, stream));
      break;
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda
