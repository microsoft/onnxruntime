/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
 * Modifications Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file default_quantb_gemm.h
 * @brief Modified from cutlass/gemm/kernel/default_gemm.h. templates for combining
 *        threadblock-scoped matrix multiply-add with  the appropriate
 *        threadblock-scoped epilogue.
 */

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass_ext/q4gemm/kernel/quantb_gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass_ext/q4gemm/threadblock/default_quantb_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "cutlass/layout/permute.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif //CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////
namespace cutlass {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for quant scales
    typename ElementQScale_,
    /// Element type for quant offsets
    typename ElementQOffset_,
    /// Layout type for quant scales and offsets
    typename LayoutQMeta_,
    /// Blocking dimensions for quantization
    typename QuantBlocking_,
    /// Access granularity of quant scales in units of elements
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute,
    ///
    typename Enable = void
>
struct DefaultQuantBGemm;

////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for quant scales
    typename ElementQScale,
    /// Element type for quant offsets
    typename ElementQOffset,
    /// Layout type for quant scales
    typename LayoutQMeta,
    /// Blocking dimensions for quantization
    typename QuantBlocking,
    /// Access granularity of quant scales in units of elements
    typename ElementC,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
>
struct DefaultQuantBGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
                         ElementQScale, ElementQOffset, LayoutQMeta, QuantBlocking,
                         ElementC, LayoutC, ElementAccumulator,
                         arch::OpClassTensorOp, arch::Sm80, ThreadblockShape, WarpShape,
                         InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages,
                         SplitKSerial, Operator, GatherA, GatherB, ScatterD,
                         PermuteDLayout, PermuteALayout, PermuteBLayout> {

  static_assert((platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value),
             "Epilogue in the kernel level must be row major");

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultQuantBMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementQScale, ElementQOffset, LayoutQMeta, QuantBlocking,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, Stages,
      Operator, false, GatherA, GatherB,
      PermuteALayout, PermuteBLayout>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using RegularEpilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount, ScatterD, PermuteDLayout>::Epilogue;

  using Affine2Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOpAffineRankN<
          2, ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  using Epilogue = typename platform::conditional<platform::is_same<LayoutC, layout::RowMajor>::value,
                                                  RegularEpilogue,
                                                  Affine2Epilogue>::type;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::QuantBGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
