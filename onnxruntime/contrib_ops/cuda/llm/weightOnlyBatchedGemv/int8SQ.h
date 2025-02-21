/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "contrib_ops/cuda/llm/common/quantization.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace onnxruntime::llm {
namespace kernels {
namespace smooth_quant {
struct Params {
  int8_t const* act;
  int8_t const* weight;
  float const* scale_tokens;
  float const* scale_channels;
  void* output;
  int m, n, k;
  onnxruntime::llm::common::QuantMode quant_mode;

  Params(int8_t const* _act, int8_t const* _weight, float const* _scale_tokens, float const* _scale_channels,
         void* _output, int _m, int _n, int _k, onnxruntime::llm::common::QuantMode _quant_mode)
      : act(_act), weight(_weight), scale_tokens(_scale_tokens), scale_channels(_scale_channels), output(_output), m(_m), n(_n), k(_k), quant_mode(_quant_mode) {
  }
};

template <typename>
void int8_sq_launcher(Params& params, cudaStream_t s);
}  // namespace smooth_quant
}  // namespace kernels
}  // namespace onnxruntime::llm
