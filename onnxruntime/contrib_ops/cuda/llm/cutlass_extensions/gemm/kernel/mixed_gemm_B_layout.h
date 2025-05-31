/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
  This file exists so that we use the same weight layout for MoE grouped gemm and regular gemm when the weight is
  quantized. The preprocessing code reads this template to know how to organize the quantized weight matrices
  to be consumed by CUTLASS.

  Note that for int4, ThreadBlockK MUST be 64.

 */

#pragma once

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/platform/platform.h"

#include "contrib_ops/cuda/llm/cutlass_extensions/arch/mma.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/tile_interleaved_layout.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename TypeA, typename TypeB, typename Arch, typename Enable = void>
struct LayoutDetailsB {
};

// Specializations for Turing+ when B is FP16. These are currently only used for MoE networks.
// TODO - Switch this to column major for weights since gemms should be more performant.
template <typename TypeA, typename Arch>
struct LayoutDetailsB<TypeA, half_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
  static constexpr int ThreadblockK = 128 * 8 / cutlass::sizeof_bits<TypeA>::value;
  using Layout = layout::ColumnMajor;
  static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<half_t>::value;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <typename TypeA, typename Arch>
struct LayoutDetailsB<TypeA, bfloat16_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
  static constexpr int ThreadblockK = 128 * 8 / cutlass::sizeof_bits<TypeA>::value;
  using Layout = layout::ColumnMajor;
  static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<bfloat16_t>::value;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <typename TypeA>
struct LayoutDetailsB<TypeA, cutlass::float_e4m3_t, arch::Sm89> {
  static constexpr int ThreadblockK = 64;

 private:
  static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint8_t>::value;
  static constexpr int ColumnsInterleaved = ElementsPerCacheLine / ThreadblockK;

 public:
  using Layout = layout::ColumnMajor;
  static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<cutlass::float_e4m3_t>::value;
  using Operator = cutlass::arch::OpMultiplyAdd;
  // for fast accumulation
  // using Operator = cutlass::arch::OpMultiplyAddFastAccum;
};

// Specializations for Turing+ when B is quantized. These can use the operator OpMultiplyAddDequantizeInterleavedBToA,
// which signals that we want to dequantize after loading from smem.
template <typename TypeA, typename Arch>
struct LayoutDetailsB<TypeA, uint8_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
  static constexpr int ThreadblockK = 128 * 8 / cutlass::sizeof_bits<TypeA>::value;

 private:
  static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint8_t>::value;
  static constexpr int ColumnsInterleaved = ElementsPerCacheLine / ThreadblockK;

 public:
  using Layout = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
  static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint8_t>::value;
  using Operator = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
};

template <typename TypeA, typename Arch>
struct LayoutDetailsB<TypeA, uint4b_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
  static constexpr int ThreadblockK = 128 * 8 / cutlass::sizeof_bits<TypeA>::value;

 private:
  static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint4b_t>::value;
  static constexpr int ColumnsInterleaved = ElementsPerCacheLine / ThreadblockK;

 public:
  using Layout = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
  static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint4b_t>::value;
  using Operator = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
