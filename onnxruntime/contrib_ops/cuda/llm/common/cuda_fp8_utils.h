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

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace onnxruntime::llm {
namespace common {

__inline__ __device__ __nv_bfloat162 fp8x2_e4m3_to_bfloat2(__nv_fp8x2_e4m3 const* in) {
  const char2 tmp_val = reinterpret_cast<char2 const*>(in)[0];
  __nv_bfloat162 out = __nv_bfloat162((float)reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.x)[0],
                                      (float)reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.y)[0]);
  return out;
}

__inline__ __device__ half2 fp8x2_e4m3_to_half2(__nv_fp8x2_e4m3 const* in) {
  const char2 tmp_val = reinterpret_cast<char2 const*>(in)[0];
  half2 out = half2((float)reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.x)[0],
                    (float)reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.y)[0]);
  return out;
}

}  // namespace common
}  // namespace onnxruntime::llm
#endif  // ENABLE_FP8
