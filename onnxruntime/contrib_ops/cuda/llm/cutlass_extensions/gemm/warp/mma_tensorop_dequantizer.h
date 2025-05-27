/*
 * Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

/*! \file
  \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/functional.h"
#include "cutlass/platform/platform.h"

#include "contrib_ops/cuda/llm/cutlass_extensions/weight_only_quant_op.h"

#include <cuda_bf16.h>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Matrix multiply operator
    typename MmaOperator_,
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of Scale elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Number of threads participating in one matrix operation
    int Threads,
    ///
    WeightOnlyQuantOp QuantOp_,
    ///
    typename Enable = void>
class MmaTensorOpDequantizer;

////////////////////////////////////////////////////////////////////////////////
// Bfloat specialization for Ampere
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    ///
    WeightOnlyQuantOp QuantOp_>
class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, bfloat16_t, layout::RowMajor, 32, QuantOp_,
                             typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >= 80 && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type> {
 public:
  /// Mma Operator
  using MmaOperator = MmaOperator_;

  // The architecture specific mma ooperator being used
  using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

  // Mma Instruction Shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  // This is the ratio of the load instruction vs the compute instruction.
  static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

  /// Type of the scales
  using ElementScale = bfloat16_t;

  /// Fragment to hold B data before Mma
  using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

  // Fragment to hold scale data to apply to B before mma
  // We need 1 fp16 per matrix iteration in the N dimension
  static constexpr int kColsPerMmaPerThread = 1;
  using FragmentScale = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;
  using FragmentZero = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

  /// Warp mma shape
  using Shape = Shape_;

  /// Layout of the scales in shared memory
  using Layout = layout::RowMajor;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<ElementScale, Layout>;

  static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales, TensorRef smem_zeros, int const warp_idx_n, int const lane_idx) {
    int const warp_offset = warp_idx_n * Shape::kN;
    int const quad = lane_idx / 4;
    int const thread_offset = warp_offset + quad;
    pointer_scale_ = smem_scales.data() + thread_offset;
    if constexpr (hasZero(QuantOp)) {
      pointer_zero_ = smem_zeros.data() + thread_offset;
    }
  }

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales, int const warp_idx_n, int const lane_idx)
      : MmaTensorOpDequantizer(smem_scales, TensorRef(), warp_idx_n, lane_idx) {
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
      scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
    }
  }

  CUTLASS_DEVICE
  void dequantize(FragmentDequantizedOperand& operand_frag, FragmentScale const& scale_frag) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
    using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
    static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn == FragmentDequantizedOperand::kElements,
                  "");

    __nv_bfloat16 const* scale_ptr = reinterpret_cast<__nv_bfloat16 const*>(&scale_frag);
    ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
      static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

      __nv_bfloat162 scalex2 = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
      __nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);

      CUTLASS_PRAGMA_UNROLL
      for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii) {
        operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);
      }
    }
#else
    // Slow path not implemented here on purpose. If we need to do HMMA on older arch, scale conversion should
    // happen before scales are stored to shared memory and we should use the fp16 dequantizer. This will avoid
    // numerous conversion instructions in GEMM main loop.
    arch::device_breakpoint();
#endif
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag, FragmentScale& zero_frag) {
    if constexpr (hasZero(QuantOp)) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
        scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
        zero_frag[mma_n_iter] = pointer_zero_[mma_n_iter * InstructionShape::kN];
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
        scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
      }
    }
  }

  CUTLASS_DEVICE
  void dequantize(
      FragmentDequantizedOperand& operand_frag, FragmentScale const& scale_frag, FragmentScale const& zero_frag) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
    using ExpandedMmaOperandB = Array<typename _MmaOperandB::Element, kExpansionFactor * _MmaOperandB::kElements>;
    static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn == FragmentDequantizedOperand::kElements,
                  "");

    __nv_bfloat16 const* scale_ptr = reinterpret_cast<__nv_bfloat16 const*>(&scale_frag);
    __nv_bfloat16 const* zero_ptr = reinterpret_cast<__nv_bfloat16 const*>(&zero_frag);

    ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
      static_assert(ExpandedMmaOperandB::kElements % 2 == 0, "");

      __nv_bfloat162 scalex2 = __bfloat162bfloat162(scale_ptr[mma_n_iter]);
      __nv_bfloat162 zerox2 = __bfloat162bfloat162(zero_ptr[mma_n_iter]);
      __nv_bfloat162* operand_bf16x2_ptr = reinterpret_cast<__nv_bfloat162*>(&operand_frag_ptr[mma_n_iter]);

      if constexpr (hasZero(QuantOp)) {
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii) {
          operand_bf16x2_ptr[ii] = __hfma2(operand_bf16x2_ptr[ii], scalex2, zerox2);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < ExpandedMmaOperandB::kElements / 2; ++ii) {
          operand_bf16x2_ptr[ii] = __hmul2(operand_bf16x2_ptr[ii], scalex2);
        }
      }
    }
#else
    // Slow path not implemented here on purpose. If we need to do HMMA on older arch, scale conversion should
    // happen before scales are stored to shared memory and we should use the fp16 dequantizer. This will avoid
    // numerous conversion instructions in GEMM main loop.
    arch::device_breakpoint();
#endif
  }

  // Adds a pointer offset in units of elements.
  CUTLASS_DEVICE
  void add_pointer_offset(int64_t const& offset) {
    static_assert(sizeof(ElementScale) > 1, "");
    pointer_scale_ += offset;
    pointer_zero_ += offset;
  }

 private:
  ElementScale const* pointer_scale_;
  ElementScale const* pointer_zero_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Turing & Ampere
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_,
    ///
    WeightOnlyQuantOp QuantOp_>
class MmaTensorOpDequantizer<MmaOperator_, Shape_, Operand::kB, half_t, layout::RowMajor, 32, QuantOp_,
                             typename platform::enable_if<MmaOperator_::ArchTag::kMinComputeCapability >= 75 && platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB, layout::ColumnMajor>::value>::type> {
 public:
  /// Mma Operator
  using MmaOperator = MmaOperator_;

  // The architecture specific mma ooperator being used
  using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

  // Mma Instruction Shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  // This is the ratio of the load instruction vs the compute instruction.
  static constexpr int kExpansionFactor = MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

  /// Type of the scales
  using ElementScale = half_t;

  /// Fragment to hold B data before Mma
  using FragmentDequantizedOperand = Array<ElementScale, MmaOperator::FragmentB::kElements>;

  // Fragment to hold scale data to apply to B before mma
  // We need 1 fp16 per matrix iteration in the N dimension
  static constexpr int kColsPerMmaPerThread = 1;
  using FragmentScale = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;
  using FragmentZero = Array<ElementScale, kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

  /// Warp mma shape
  using Shape = Shape_;

  /// Layout of the scales in shared memory
  using Layout = layout::RowMajor;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<ElementScale, Layout>;

  static constexpr WeightOnlyQuantOp QuantOp = QuantOp_;

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales, TensorRef smem_zeros, int const warp_idx_n, int const lane_idx) {
    int const warp_offset = warp_idx_n * Shape::kN;
    int const quad = lane_idx / 4;
    int const thread_offset = warp_offset + quad;
    pointer_scale_ = smem_scales.data() + thread_offset;
    if constexpr (hasZero(QuantOp)) {
      pointer_zero_ = smem_zeros.data() + thread_offset;
    }
  }

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales, int const warp_idx_n, int const lane_idx)
      : MmaTensorOpDequantizer(smem_scales, TensorRef(), warp_idx_n, lane_idx) {
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
      scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
    }
  }

  CUTLASS_DEVICE
  void dequantize(FragmentDequantizedOperand& operand_frag, FragmentScale const& scale_frag) {
    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
    using ExpandedMmaOperandB = Array<typename FragmentDequantizedOperand::Element, kExpansionFactor * _MmaOperandB::kElements>;
    static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn == FragmentDequantizedOperand::kElements,
                  "");

    multiplies<ExpandedMmaOperandB> mul_op;

    ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
      operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
    }
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag, FragmentScale& zero_frag) {
    if constexpr (hasZero(QuantOp)) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
        scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
        zero_frag[mma_n_iter] = pointer_zero_[mma_n_iter * InstructionShape::kN];
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
        scale_frag[mma_n_iter] = pointer_scale_[mma_n_iter * InstructionShape::kN];
      }
    }
  }

  CUTLASS_DEVICE
  void dequantize(
      FragmentDequantizedOperand& operand_frag, FragmentScale const& scale_frag, FragmentScale const& zero_frag) {
    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
    using ExpandedMmaOperandB = Array<typename FragmentDequantizedOperand::Element, kExpansionFactor * _MmaOperandB::kElements>;
    static_assert(ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn == FragmentDequantizedOperand::kElements,
                  "");

    multiplies<ExpandedMmaOperandB> mul_op;
    ExpandedMmaOperandB* operand_frag_ptr = reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);

    if constexpr (hasZero(QuantOp)) {
      plus<ExpandedMmaOperandB> plus_op;

      CUTLASS_PRAGMA_UNROLL
      for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
        operand_frag_ptr[mma_n_iter] = plus_op(mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]), zero_frag[mma_n_iter]);
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn; ++mma_n_iter) {
        operand_frag_ptr[mma_n_iter] = mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
      }
    }
  }

  // Adds a pointer offset in units of elements.
  CUTLASS_DEVICE
  void add_pointer_offset(int64_t const& offset) {
    static_assert(sizeof(ElementScale) > 1, "");
    pointer_scale_ += offset;
    pointer_zero_ += offset;
  }

 private:
  ElementScale const* pointer_scale_;
  ElementScale const* pointer_zero_;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
