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
 * @file default_quantb_mma_core.h
 * @brief Modified from cutlass/gemm/threadblock/default_mma_core.h.
 *        Defining data layout in shared memory, and its iterators.
 */

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass_ext/q4gemm/warp/default_quantb_mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass_ext/q4gemm/threadblock/optional_regular_tile_access_iter.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Element data type of quant scale
    typename ElementQScale,
    /// Element data type of quant offset
    typename ElementQOffset,
    /// Layout of quant scale
    typename LayoutQMeta,
    /// Blocking dimensions for quantization
    typename QuantBlocking,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    /// Number of stages
    int Stages = 2,
    /// Operation performed by MMA
    typename Operator = typename platform::conditional<
        (platform::is_same<OperatorClass,
                           cutlass::arch::OpClassTensorOp>::value) &&
            (platform::is_same<ElementA, int8_t>::value ||
             platform::is_same<ElementA, int4b_t>::value ||
             platform::is_same<ElementA, uint8_t>::value ||
             platform::is_same<ElementA, uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA =
        cutlass::arch::CacheOperation::Global,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB =
        cutlass::arch::CacheOperation::Global,
    /// per-element transformation for elements of A
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// per-element transformation for elements of B
    ComplexTransform TransformB = ComplexTransform::kNone,
    bool IsComplex = false // (is_complex<ElementA>::value || is_complex<ElementB>::value)
>
struct DefaultQuantBMmaCore;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: column-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Element data type of quant scale
    typename ElementQScale_,
    /// Element data type of quant offset
    typename ElementQOffset_,
    /// Layout of quant scale
    typename LayoutQMeta_,
    /// Blocking dimensions for quantization
    typename QuantBlocking_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultQuantBMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementQScale_, ElementQOffset_, LayoutQMeta_, QuantBlocking_,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                      Operator_, false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;

  using ElementQScale = ElementQScale_;
  using ElementQOffset = ElementQOffset_;
  using LayoutQMeta = LayoutQMeta_;
  using QuantBlocking = QuantBlocking_;

  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN,
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      (Shape::kK / 2) / (kAccessSizeInBits / sizeof_bits<ElementB>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK>;

  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, Shape::kK/2>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK/2, Shape::kN/2>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK/2, Shape::kN/2>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  using SmemLayoutQScale = LayoutQMeta;
  using SmemLayoutQOffset = LayoutQMeta;

  /// Threadblock-level quantization meta data shape
  using ThreadblockQShape = MatrixShape<Shape::kK / QuantBlocking::kRow, Shape::kN / QuantBlocking::kColumn>;
  static_assert(Shape::kK % QuantBlocking::kRow == 0, "K must be multiple of QuantBlocking::kRow");
  static_assert(Shape::kN % QuantBlocking::kColumn == 0, "N must be multiple of QuantBlocking::kColumn");
  static_assert(ThreadblockQShape::kCount > 0, "QuantBlocking too big to fit in a thread block!");
  static_assert(QuantBlocking::kRow == 1 || QuantBlocking::kColumn == 1,
        "Only support single column or row quantize blocking!");
  static_assert(QuantBlocking::kColumn != 1 || std::is_same<LayoutQMeta, layout::RowMajor>::value,
        "Quant scale matrix's major dimension must have more elements, to facilitate fast loading!");

  /// Threadblock-level quantization meta data shape in pitch-linear layout
  using TBQPitchLinearShape = typename std::conditional<
      std::is_same<LayoutQMeta, layout::RowMajor>::value,
      layout::PitchLinearShape<ThreadblockQShape::kColumn, ThreadblockQShape::kRow>,
      layout::PitchLinearShape<ThreadblockQShape::kRow, ThreadblockQShape::kColumn>>::type;

  /// By default we would like to use 128b load. However, we can't load more than
  /// a column at a time in a column major layout.
  static int const kElementsPerAccessQScale =
      (kAccessSizeInBits / sizeof_bits<ElementQScale>::value) > TBQPitchLinearShape::kContiguous
          ? TBQPitchLinearShape::kContiguous
          : (kAccessSizeInBits / sizeof_bits<ElementQScale>::value);

  /// quant scale is tiny.  Not all threads are needed.
  static int const kAccessCntQScale = ThreadblockQShape::kCount / kElementsPerAccessQScale;
  static int const kThreadsQScale = (kAccessCntQScale > kThreads) ? kThreads : kAccessCntQScale;

  using IteratorThreadMapQScale = transform::PitchLinearStripminedThreadMap<
      TBQPitchLinearShape, kThreadsQScale, kElementsPerAccessQScale>;

  using SmemIteratorQScale = transform::threadblock::RegularTileAccessIterator<
        ThreadblockQShape, ElementQScale, SmemLayoutQScale, 1, IteratorThreadMapQScale>;

  static int const kElementsPerAccessQOffset =
      (kAccessSizeInBits / sizeof_bits<ElementQOffset>::value) > TBQPitchLinearShape::kContiguous
          ? TBQPitchLinearShape::kContiguous
          : (kAccessSizeInBits / sizeof_bits<ElementQOffset>::value);
  static int const kAccessCntQOffset = ThreadblockQShape::kCount / kElementsPerAccessQOffset;
  static int const kThreadsQOffset = (kAccessCntQOffset > kThreads) ? kThreads : kAccessCntQOffset;

  using IteratorThreadMapQOffset = transform::PitchLinearStripminedThreadMap<
      TBQPitchLinearShape, kThreadsQOffset, kElementsPerAccessQOffset>;

  using SmemIteratorQOffset = transform::threadblock::OptionalRegularTileAccessIterator<
        ThreadblockQShape, ElementQOffset, SmemLayoutQOffset, 1, IteratorThreadMapQOffset, kThreads>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultQuantBMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementQScale, SmemLayoutQScale, ElementQOffset, SmemLayoutQScale, QuantBlocking,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
