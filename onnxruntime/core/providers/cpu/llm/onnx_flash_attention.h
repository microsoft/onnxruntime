// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/llm/attention_parameters.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

bool CanUseOnnxFlashAttention(const attention_helper::AttentionParameters& parameters,
                              const Tensor* mask_index,
                              const Tensor* past_key,
                              const Tensor* past_value,
                              const Tensor* present_key,
                              const Tensor* present_value,
                              const Tensor* output_qk);

Status DispatchOnnxFlashAttention(const float* Q,
                                  const float* K,
                                  const float* V,
                                  const Tensor* mask_index,
                                  float* output,
                                  const attention_helper::AttentionParameters& parameters,
                                  concurrency::ThreadPool* tp);

}  // namespace onnxruntime
