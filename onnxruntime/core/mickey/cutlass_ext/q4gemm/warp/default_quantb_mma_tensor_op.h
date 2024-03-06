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
 * @file default_quantb_mma_tensor_op.h
 * @brief Modified from cutlass/gemm/warp/default_mma_tensor_op.h
 * Default warp-level GEMM operators selected by data type, size, and layouts of operands.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass_ext/q4gemm/warp/quantb_mma_tensor_op.h"

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for m-by-n-by-kgroup
template <
    /// Shape of one matrix production operation (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A elements
    typename ElementA,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA,
    /// Data type of B elements
    typename ElementB,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB,
    /// Data type of quant scales
    typename ElementQScale,
    /// Layout of quant scales (concept: MatrixLayout)
    typename SmemLayoutQScale,
    /// Data type of quant offsets
    typename ElementQOffset,
    /// Layout of quant offsets (concept: MatrixLayout)
    typename SmemLayoutQOffset,
    /// Blocking size of quantization
    typename QuantBlocking,
    /// Element type of C matrix
    typename ElementC,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC,
    /// Operator describing the tensor operation
    typename Operator_ = arch::OpMultiplyAdd,
    /// Number of partitions along K dimension
    int PartitionsK = 1,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false>
struct DefaultQuantBMmaTensorOp {
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<InstructionShape_, 32, ElementA,
                         cutlass::layout::RowMajor, ElementB,
                         cutlass::layout::ColumnMajor, ElementC,
                         cutlass::layout::RowMajor, Operator_>,
      cutlass::MatrixShape<1, 1> >;

  // Define the warp-level tensor op
  using Type = cutlass::gemm::warp::QuantBMmaTensorOp<
      WarpShape_, ElementA, LayoutA, ElementB, LayoutB, ElementQScale, SmemLayoutQScale,
      ElementQOffset, SmemLayoutQOffset, QuantBlocking, ElementC, LayoutC,
      Policy, PartitionsK, AccumulatorsInRowMajor>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
