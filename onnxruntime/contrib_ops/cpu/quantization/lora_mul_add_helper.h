// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"

namespace onnxruntime::contrib {

inline constexpr int64_t kLoraBlockSize = 32;

inline Status CheckLoraMulAddInputs(const Tensor* base,
                                    const Tensor* input,
                                    const Tensor* lora_a,
                                    const Tensor* lora_b,
                                    const Tensor* scale_a,
                                    const Tensor* scale_b) {
  ORT_RETURN_IF_NOT(base && input && lora_a && lora_b && scale_a && scale_b,
                    "LoraMulAdd requires base, activation, INT8 weights, and FP32 scales.");
  ORT_RETURN_IF_NOT((input->IsDataType<float>() || input->IsDataType<MLFloat16>()) &&
                        base->DataType() == input->DataType() &&
                        lora_a->IsDataType<int8_t>() && lora_b->IsDataType<int8_t>() &&
                        scale_a->IsDataType<float>() && scale_b->IsDataType<float>(),
                    "LoraMulAdd requires matching FP32/FP16 base and activation, INT8 weights, and FP32 scales.");
  const auto& shape = input->Shape();
  const auto& base_shape = base->Shape();
  const auto& a_shape = lora_a->Shape();
  const auto& b_shape = lora_b->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() > 0 &&
                        base_shape.NumDimensions() == shape.NumDimensions() &&
                        a_shape.NumDimensions() == 2 && b_shape.NumDimensions() == 2,
                    "LoraMulAdd requires matching activation/base ranks and matrix LoRA weights.");
  const int64_t K = shape[shape.NumDimensions() - 1];
  const int64_t N = base_shape[base_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(K > 0 && N > 0 &&
                        a_shape[0] == K && b_shape[1] == N && a_shape[1] == b_shape[0],
                    "LoraMulAdd requires X[..., K], base[..., N], Q_A[K, rank], and Q_B[rank, N].");
  for (size_t axis = 0; axis + 1 < shape.NumDimensions(); ++axis) {
    ORT_RETURN_IF_NOT(shape[axis] == base_shape[axis], "LoraMulAdd base batch dimensions must match X.");
  }
  const auto& sa_shape = scale_a->Shape();
  const auto& sb_shape = scale_b->Shape();
  const int64_t rank = a_shape[1];
  ORT_RETURN_IF_NOT(sa_shape.NumDimensions() == 2 && sb_shape.NumDimensions() == 2 &&
                        sa_shape[0] == (K - 1) / kLoraBlockSize + 1 && sa_shape[1] == rank &&
                        sb_shape[0] == rank / kLoraBlockSize + (rank % kLoraBlockSize != 0) &&
                        sb_shape[1] == N,
                    "LoraMulAdd scales must match axis-0 blocked quantization with block_size=32.");
  return Status::OK();
}

}  // namespace onnxruntime::contrib
