/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
/*
  This file exists so that we use the same weight layout for MoE grouped gemm and regular gemm when the weight is
  quantized. The preprocessing code reads this template to know how to organize the quantized weight matrices
  to be consumed by CUTLASS.

  Note that for int4, ThreadBlockK MUST be 64.

 */

#ifdef USE_CUTLASS

#pragma once

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/platform/platform.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename TypeB, typename Arch, typename Enable = void>
struct LayoutDetailsB {};

// Volta specialiations. Volta will dequantize before STS, so we need a different operator
template <typename TypeB>
struct LayoutDetailsB<TypeB, arch::Sm70> {
  static constexpr int ThreadblockK = 64;
  using Layout = layout::RowMajor;
  static constexpr int ElementsPerAccess = 8;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Specializations for Turing+ when B is FP16. These are currently only used for MoE networks.
// TODO - Switch this to column major for weights since gemms should be more performant.
template <typename Arch>
struct LayoutDetailsB<half_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
  static constexpr int ThreadblockK = 64;
  using Layout = layout::RowMajor;
  static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<half_t>::value;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <typename TypeA, typename TypeB, typename arch, typename Enable = void>
struct MixedGemmArchTraits {};

template <typename arch>
struct MixedGemmArchTraits<float, float, arch> {
  static constexpr int Stages = 2;
  using OperatorClass = cutlass::arch::OpClassSimt;
  using AccType = float;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr int ElementsPerAccessA = 1;
  static constexpr int ElementsPerAccessB = 1;
  static constexpr int ElementsPerAccessC = 1;
  static constexpr int ThreadblockK = 8;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Operator = cutlass::arch::OpMultiplyAdd;
};

// ========================= Volta Traits ===========================
// Volta will always dequantize after the global memory load.
// This will instantiate any HMMA tensorcore kernels for Volta.
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA, TypeB, cutlass::arch::Sm70,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value>::type> {
 private:
  using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm70>;

 public:
  static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = typename LayoutDetails::Layout;

  static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
  static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
  static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  using Operator = typename LayoutDetails::Operator;
};

// ======================= Turing Traits ==============================
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA, TypeB, cutlass::arch::Sm75,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value>::type> {
 private:
  using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm75>;

 public:
  static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = typename LayoutDetails::Layout;

  static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
  static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
  static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

  using Operator = typename LayoutDetails::Operator;
};

// ======================= Ampere Traits ==============================
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<
    TypeA, TypeB, cutlass::arch::Sm80,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value>::type> {
 private:
  using LayoutDetails = LayoutDetailsB<TypeB, cutlass::arch::Sm80>;

 public:
  static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using AccType = float;
  using LayoutB = typename LayoutDetails::Layout;

  static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
  static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
  static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using Operator = typename LayoutDetails::Operator;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

#endif
