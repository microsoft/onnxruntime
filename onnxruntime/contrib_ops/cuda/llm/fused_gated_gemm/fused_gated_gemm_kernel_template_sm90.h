/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/collective/collective_builder_gated.hpp"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/gemm_universal_gated.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif  // __GNUC__

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {
using namespace cute;

template <typename ElementType, typename AccumElementType, typename CTAShape, typename ClusterShape,
          typename MainloopScheduleType, typename EpilogueScheduleType, typename TileSchedulerType = void,
          template <class /* ElementCompute */> class Activation = cutlass::epilogue::thread::SiLu, bool SwapAB = false>
struct DeviceGemmGatedSm90 {
  static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

  // A matrix configuration
  using ElementA = ElementType;                                                   // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;                                      // Layout type for A matrix operand
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                                                  // matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = ElementType;                                                   // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;                                   // Layout type for B matrix operand
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                                                  // matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementC = ElementType;  // Element type for C matrix operands
  // using LayoutC = cutlass::layout::RowMajor;         // Layout type for C matrix operands
  using LayoutC = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C matrices in units of
                                                                                  // elements (up to 16 bytes)

  // Output matrix configuration
  using ElementOutput = ElementType;  // Element type for output matrix operands
  // using LayoutOutput = cutlass::layout::RowMajor; // Layout type for output matrix operands
  using LayoutOutput = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator = AccumElementType;           // Element type for internal accumulation
  using ElementCompute = float;                          // Element type for compute
  using ArchTag = cutlass::arch::Sm90;                   // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = CTAShape;                            // Threadblock-level tile size
  using KernelSchedule = MainloopScheduleType;
  using EpilogueSchedule = EpilogueScheduleType;
  using TileScheduler = TileSchedulerType;

  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using FusionOperation = cutlass::epilogue::fusion::ScaledAcc<ElementOutput, ElementCompute>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
                                                                                       TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator, ElementC, LayoutC,
                                                                                       AlignmentC, ElementOutput, LayoutOutput, AlignmentOutput, EpilogueSchedule, FusionOperation>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderGated<ArchTag, OperatorClass,
                                                                                        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
                                                                                        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                                                                                            sizeof(typename CollectiveEpilogue::SharedStorage))>,
                                                                                        KernelSchedule, Activation, SwapAB>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversalGated<Shape<int, int, int, int>,  // Indicates ProblemShape
                                                               CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
