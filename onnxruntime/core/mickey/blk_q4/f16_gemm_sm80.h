/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *   blk_q4/f16_gemm_sm80.h
 *
 * Abstract:
 *   Entry point for Q4F16 GEMM kernel for SM80 devices.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass_ext/q4gemm/device/quantb_gemm.h"

namespace onnxruntime {
namespace cuda {

//
// This is the implementation of the quantized GEMM kernel for 16b float x blocked quantized 4b data type
//
template <
    typename ElementDequant_,  // <- data type of dequantized elements for gemm, fp16 or bf16
    typename QuantBlocking_,   // <- weights block per scale, cutlass::MatrixShape<x,y>
    bool SmallM,               // <- true if M <= 16
    bool kHasQuantOffset>
struct BlkQ4F16GemmImpl {
  //
  // Type definitions
  //

  using ElementDequant = ElementDequant_;
  using QuantBlocking = QuantBlocking_;

  static_assert(sizeof(ElementDequant) == 2, "q4f16gemm kerenl only support 16b operands!");

  // Data types that are fixed for this kernel
  using ElementAccumulator = float;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementInputA = ElementDequant;
  using ElementOutput = ElementDequant;

  using ElementW = uint8_t;  // <- Weight is int4, uint8 for two of them

  // We pack 4 weights into one 16b element, so as to leverage cutlass tile iterators
  // for async shared memory loading and minimize bank conflict
  using ElementWPack = ElementDequant;

  using ElementQScale = ElementDequant;  // <- data type of quantization scale
  using ElementQOffset = uint8_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputWPack = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // Layout of quantization scale and offset, oriented to be loaded using less instructions
  // in a warp tile
  using LayoutInputQScale =
      typename std::conditional<QuantBlocking::kRow == 1,
                                cutlass::layout::ColumnMajor,
                                cutlass::layout::RowMajor>::type;  // <- layout of quantization scale

  using ShapeMMAThreadBlock =
      typename std::conditional<SmallM,
                                cutlass::gemm::GemmShape<16, 64, 64>,
                                cutlass::gemm::GemmShape<128, 256, 64>>::type;

  static constexpr int MinN = QuantBlocking::kColumn > 32 ? QuantBlocking::kColumn : 32;
  using ShapeMMAWarp =
      typename std::conditional<SmallM,
                                cutlass::gemm::GemmShape<16, MinN, 64>,
                                cutlass::gemm::GemmShape<64, 64, 64>>::type;

  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

  // This code section describes the epilogue part of the kernel
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,                                     // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                         // memory access. For a byte, it's 16
                                                         // elements. This becomes the vector width of
                                                         // math instructions in the epilogue too
      ElementAccumulator,                                // <- data type of accumulator
      ElementComputeEpilogue>;                           // <- data type for alpha/beta in linear combination function

  // Number of pipelines you want to use
  static constexpr int NumStages = 3;

  using Gemm = cutlass::gemm::device::QuantBGemm<
      ElementInputA,
      LayoutInputA,
      ElementWPack,
      LayoutInputWPack,
      ElementQScale,
      typename std::conditional<kHasQuantOffset, ElementQOffset, std::monostate>::type,
      LayoutInputQScale,
      QuantBlocking,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages>;

  using Arguments = typename Gemm::Arguments;

  // Invoke gemm kernel (the version with quantization offset)
  static cutlass::Status run(
      cudaStream_t stream,
      const cutlass::gemm::GemmCoord& problem_size_,
      cutlass::TensorRef<ElementInputA const, LayoutInputA> ref_A_,
      cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_B_,
      cutlass::TensorRef<ElementQScale const, LayoutInputQScale> ref_Qscale_,
      cutlass::TensorRef<ElementQOffset const, LayoutInputQScale> ref_Qoffset_,
      cutlass::TensorRef<ElementOutput const, LayoutOutput> ref_C_,
      cutlass::TensorRef<ElementOutput, LayoutOutput> ref_D_,
      typename EpilogueOp::Params epilogue_ = typename EpilogueOp::Params()) {
    if constexpr (!kHasQuantOffset) {
      return cutlass::Status::kErrorNotSupported;
    } else {
      if constexpr (ShapeMMAThreadBlock::kM == 16) {
        if (problem_size_.m() > 16) {
          // For M > 16, the caller should have picked the
          // kernel with bigger M
          return cutlass::Status::kErrorNotSupported;
        }
      }

      // Construct Gemm arguments
      Arguments args{
          problem_size_,
          ref_A_,
          ref_B_,
          ref_Qscale_,
          ref_Qoffset_,
          ref_C_,
          ref_D_,
          epilogue_};

      Gemm gemm_op;

      // Check if this GEMM can be run or not
      cutlass::Status status = gemm_op.can_implement(args);
      if (status != cutlass::Status::kSuccess) {
        return status;
      }

      // Launch the CUTLASS GEMM kernel.
      return gemm_op(args, nullptr, stream);
    }
  }

  // Invoke gemm kernel (the version without quantization offset)
  static cutlass::Status run(
      cudaStream_t stream,
      const cutlass::gemm::GemmCoord& problem_size_,
      cutlass::TensorRef<ElementInputA const, LayoutInputA> ref_A_,
      cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_B_,
      cutlass::TensorRef<ElementQScale const, LayoutInputQScale> ref_Qscale_,
      cutlass::TensorRef<ElementOutput const, LayoutOutput> ref_C_,
      cutlass::TensorRef<ElementOutput, LayoutOutput> ref_D_,
      typename EpilogueOp::Params epilogue_ = typename EpilogueOp::Params()) {
    if constexpr (kHasQuantOffset) {
      return cutlass::Status::kErrorNotSupported;
    } else {
      if constexpr (ShapeMMAThreadBlock::kM == 16) {
        if (problem_size_.m() > 16) {
          // For M > 16, the caller should have picked the
          // kernel with bigger M
          return cutlass::Status::kErrorNotSupported;
        }
      }

      // Construct Gemm arguments
      Arguments args{
          problem_size_,
          ref_A_,
          ref_B_,
          ref_Qscale_,
          ref_C_,
          ref_D_,
          epilogue_};

      Gemm gemm_op;

      // Check if this GEMM can be run or not
      cutlass::Status status = gemm_op.can_implement(args);
      if (status != cutlass::Status::kSuccess) {
        return status;
      }

      // Launch the CUTLASS GEMM kernel.
      return gemm_op(args, nullptr, stream);
    }
  }
};

}  // namespace cuda
}  // namespace onnxruntime
