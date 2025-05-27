/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/llm/nv_infer_datatype.h"

#include "cutlass/half.h"
#include <cuda_fp16.h>

#include "cutlass/bfloat16.h"
#include <cuda_bf16.h>

#include "cutlass/float8.h"
#include <cuda_fp8.h>

#if defined(ENABLE_FP4)
#include "cutlass/float_subbyte.h"
#include <cuda_fp4.h>
#endif

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {
///////////////////////////////////////////////////////////////////////////////////////////////////
// nvinfer::DataType to Cutlass
///////////////////////////////////////////////////////////////////////////////////////////////////
template <nvinfer::DataType>
struct CutlassType {
  using type = void;
};

template <>
struct CutlassType<nvinfer::DataType::kHALF> {
  using type = cutlass::half_t;
};

template <>
struct CutlassType<nvinfer::DataType::kBF16> {
  using type = cutlass::bfloat16_t;
};

template <>
struct CutlassType<nvinfer::DataType::kFP8> {
  using type = cutlass::float_e4m3_t;
};

#if defined(ENABLE_FP4)
template <>
struct CutlassType<nvinfer::DataType::kFP4> {
  using type = cutlass::float_e2m1_t;
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA to Cutlass

template <typename T>
struct CudaToCutlassTypeAdapter {
  using type = T;
};

template <>
struct CudaToCutlassTypeAdapter<half> {
  using type = cutlass::half_t;
};

template <>
struct CudaToCutlassTypeAdapter<__nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

#if defined(ENABLE_FP8)
template <>
struct CudaToCutlassTypeAdapter<__nv_fp8_e4m3> {
  using type = cutlass::float_e4m3_t;
};

template <>
struct CudaToCutlassTypeAdapter<__nv_fp8_e5m2> {
  using type = cutlass::float_e5m2_t;
};
#endif

#if defined(ENABLE_FP4)
template <>
struct CudaToCutlassTypeAdapter<__nv_fp4_e2m1> {
  using type = cutlass::float_e2m1_t;
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// Cutlass to CUDA

template <typename T>
struct CudaToCudaTypeAdapter {
  using type = T;
};

template <>
struct CudaToCudaTypeAdapter<cutlass::half_t> {
  using type = half;
};

template <>
struct CudaToCudaTypeAdapter<cutlass::bfloat16_t> {
  using type = __nv_bfloat16;
};

#if defined(ENABLE_FP8)
template <>
struct CudaToCudaTypeAdapter<cutlass::float_e4m3_t> {
  using type = __nv_fp8_e4m3;
};

template <>
struct CudaToCudaTypeAdapter<cutlass::float_e5m2_t> {
  using type = __nv_fp8_e5m2;
};
#endif

#if defined(ENABLE_FP4)
template <>
struct CudaToCudaTypeAdapter<cutlass::float_e2m1_t> {
  using type = __nv_fp4_e2m1;
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
