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
 * @file quantb_gemm.h
 * @brief Modified from cutlass/gemm/device/gemm.h, boilerplate code passing input pointers to the kernel.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm.h"

#include "cutlass_ext/q4gemm/kernel/default_quantb_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/layout/permute.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*! A specialized GEMM operator for quantized B GEMM.

  It is modified from cutlass::gemm::device::Gemm. Both this class and the original Gemm class
  are pretty much boilerplate code that construct the Gemm kernel class, and pass parameters
  and controls to it. The only difference is that this class has a few more template parameters
  to support quantization.

  This implementation pretty much follows the design of cutlass. But this class seems to be
  just a wrapper of the Gemm kernel class. Consider combining them in future iterations.

*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for quant scales
    typename ElementQScale_,
    /// Element type for quant offsets
    typename ElementQOffset_,
    /// Layout type for quant scales and offsets
    typename LayoutQMeta_,
    /// Blocking dimensions for quantization
    typename QuantBlocking_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm80,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute>
class QuantBGemm {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  // Quantization Parameters
  static_assert(std::is_same<LayoutB, layout::ColumnMajor>::value,
                "LayoutB, i.e. packed weights must appear ColumnMajor.");
  static_assert(InstructionShape::kK == 16,
                "InstructionShape::kK must be a multiple of 16 (2 tiles), required by 4b weight packing layout.");
  using ElementQScale = ElementQScale_;
  using ElementQOffset = ElementQOffset_;
  using LayoutQMeta = LayoutQMeta_;
  using QuantBlocking = QuantBlocking_;
  static constexpr bool kHasQOffset = !(std::is_same<ElementQOffset, std::monostate>::value);

  // TODO(chenfucn): consider moving to uint4_t or smaller for QOffset
  static_assert(!kHasQOffset || std::is_same<ElementQOffset_, uint8_t>::value, "QOffset must be uint8_t");

  /// Define the kernel
  using GemmKernel = typename kernel::DefaultQuantBGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementQScale,
    ElementQOffset,
    LayoutQMeta,
    QuantBlocking,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial,
    Operator,
    GatherA,
    GatherB,
    ScatterD,
    PermuteDLayout
  >::GemmKernel;

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    TensorRef<ElementQScale const, LayoutQMeta> ref_Qscale;
    TensorRef<ElementQOffset const, LayoutQMeta> ref_Qoffset;

    typename EpilogueOutputOp::Params epilogue;

    // split-K parallelism (etc.) are not yet supported, keeping this for future extension
    int split_k_slices{1};
    // For gather+scatter operations
    int const *gather_A_indices{nullptr};
    int const *gather_B_indices{nullptr};
    int const *scatter_D_indices{nullptr};

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0) {}

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementQScale const, LayoutQMeta> ref_Qscale_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params()):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_Qscale(ref_Qscale_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_) {
        assert(!kHasQOffset);
    }

    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementQScale const, LayoutQMeta> ref_Qscale_,
      TensorRef<ElementQOffset const, LayoutQMeta> ref_Qoffset_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params()):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_Qscale(ref_Qscale_),
      ref_Qoffset(ref_Qoffset_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_) {
        assert(kHasQOffset);
    }
  };

 private:
  /// Kernel parameters object
  typename GemmKernel::Params params_;

 public:
  /// Constructs the GEMM.
  QuantBGemm() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = GemmKernel::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_Qscale.non_const_ref(),
      args.ref_Qoffset.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    size_t bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (kSplitKSerial && args.split_k_slices > 1) {

      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (kSplitKSerial) {
      if (args.split_k_slices > 1) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }

        size_t bytes = get_workspace_size(args);

        cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    } else {

      if (args.split_k_slices > 1) {
        return Status::kErrorInvalidProblem;
      }
    }

    // Initialize the Params structure
    params_ = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_Qscale.non_const_ref(),
      args.ref_Qoffset.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    if (kSplitKSerial && args.split_k_slices > 1) {
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    params_.ref_A.reset(args.ref_A.non_const_ref().data());
    params_.ref_B.reset(args.ref_B.non_const_ref().data());
    params_.ref_Qscale.reset(args.ref_Qscale.non_const_ref().data());
    params_.ref_Qoffset.reset(args.ref_Qoffset.non_const_ref().data());
    params_.ref_C.reset(args.ref_C.non_const_ref().data());
    params_.ref_D.reset(args.ref_D.data());
    params_.output_op = args.epilogue;
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        std::cerr << "Failed to obtain maximum shared memory size " << smem_size << " for kernel: "
                  << cudaGetErrorString(result) << "\n";
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};


////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
