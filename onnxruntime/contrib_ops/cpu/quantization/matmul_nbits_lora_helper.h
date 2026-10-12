// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"

namespace onnxruntime::contrib {

inline constexpr int64_t kLoraBlockSize = 32;

inline Status CheckMatMulNBitsLoraInputs(const Tensor* input,
                                         const Tensor* lora_a,
                                         const Tensor* lora_b,
                                         const Tensor* scale_a,
                                         const Tensor* scale_b,
                                         int64_t K, int64_t N) {
  ORT_RETURN_IF_NOT(input && lora_a && lora_b && scale_a && scale_b,
                    "MatMulNBitsLora requires activation, INT8 weights, and FP32 scales.");
  ORT_RETURN_IF_NOT((input->IsDataType<float>() || input->IsDataType<MLFloat16>()) &&
                        lora_a->IsDataType<int8_t>() && lora_b->IsDataType<int8_t>() &&
                        scale_a->IsDataType<float>() && scale_b->IsDataType<float>(),
                    "MatMulNBitsLora requires FP32/FP16 activation, INT8 weights, and FP32 scales.");
  const auto& shape = input->Shape();
  const auto& a_shape = lora_a->Shape();
  const auto& b_shape = lora_b->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() > 0 &&
                        a_shape.NumDimensions() == 2 && b_shape.NumDimensions() == 2,
                    "MatMulNBitsLora requires activation rank >= 1 and matrix LoRA weights.");
  ORT_RETURN_IF_NOT(K > 0 && N > 0 &&
                        shape[shape.NumDimensions() - 1] == K &&
                        a_shape[0] == K && b_shape[1] == N && a_shape[1] == b_shape[0],
                    "MatMulNBitsLora requires X[..., K], Q_A[K, rank], and Q_B[rank, N].");
  const auto& sa_shape = scale_a->Shape();
  const auto& sb_shape = scale_b->Shape();
  const int64_t rank = a_shape[1];
  ORT_RETURN_IF_NOT(sa_shape.NumDimensions() == 2 && sb_shape.NumDimensions() == 2 &&
                        sa_shape[0] == (K - 1) / kLoraBlockSize + 1 && sa_shape[1] == rank &&
                        sb_shape[0] == rank / kLoraBlockSize + (rank % kLoraBlockSize != 0) &&
                        sb_shape[1] == N,
                    "MatMulNBitsLora scales must match axis-0 blocked quantization with block_size=32.");
  return Status::OK();
}

}  // namespace onnxruntime::contrib
