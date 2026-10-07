// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"

namespace onnxruntime::contrib {

inline Status CheckMatMulNBitsLoraInputs(const Tensor* input,
                                         const Tensor* lora_a,
                                         const Tensor* lora_b,
                                         int64_t K,
                                         int64_t N) {
  ORT_RETURN_IF_NOT(input && lora_a && lora_b,
                    "MatMulNBitsLora requires activation and both LoRA tensors.");
  ORT_RETURN_IF_NOT((input->IsDataType<float>() || input->IsDataType<MLFloat16>()) &&
                        lora_a->DataType() == input->DataType() &&
                        lora_b->DataType() == input->DataType(),
                    "MatMulNBitsLora requires matching FP32 or FP16 tensors.");
  const auto& shape = input->Shape();
  const auto& a_shape = lora_a->Shape();
  const auto& b_shape = lora_b->Shape();
  ORT_RETURN_IF_NOT(K > 0 && N > 0 && shape.NumDimensions() > 0 &&
                        shape[shape.NumDimensions() - 1] == K &&
                        a_shape.NumDimensions() == 2 && b_shape.NumDimensions() == 2 &&
                        a_shape[0] == K && b_shape[1] == N && a_shape[1] == b_shape[0],
                    "MatMulNBitsLora requires A[..., K], lora_A[K, rank], and lora_B[rank, N].");
  return Status::OK();
}

}  // namespace onnxruntime::contrib
