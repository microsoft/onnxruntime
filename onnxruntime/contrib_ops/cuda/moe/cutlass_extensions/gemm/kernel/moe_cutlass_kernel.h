/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief
*/

#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"

#include "contrib_ops/cuda/moe/cutlass_extensions/gemm/kernel/gemm_moe_problem_visitor.h"
#include "contrib_ops/cuda/moe/cutlass_extensions/tile_interleaved_layout.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
// This section exists to that we can use the same kernel code for regular gemm and dequantizing gemms.
// It will dispatch to the dequantizing gemm if the Mma type has an Iterator for scales in global.
template <typename...>
using void_t = void;

template <typename Mma, typename = void>
struct use_dq_gemm : platform::false_type {};

template <typename Mma>
struct use_dq_gemm<Mma, void_t<typename Mma::IteratorScale>> : platform::true_type {};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,                        ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,                   ///! Epilogue
          typename ThreadblockSwizzle_,         ///! Threadblock swizzling function
          typename KernelArch,                  ///! The Architecture this kernel is compiled for. Used since SIMT
                                                /// kernels lose top-level arch.
          GroupScheduleMode GroupScheduleMode_  ///! Type of scheduling to perform
          >
struct MoeFCGemm {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = false;

  // Optional transpose
  using MapArguments =
      kernel::detail::MapArguments<typename Mma::IteratorA::Element, typename Mma::IteratorA::Layout, Mma::kTransformA,
                                   Mma::IteratorA::AccessType::kElements, typename Mma::IteratorB::Element,
                                   typename Mma::IteratorB::Layout, Mma::kTransformB,
                                   Mma::IteratorB::AccessType::kElements, typename Mma::LayoutC, kTransposed>;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion.
  static_assert(!kTransposed, "Transpose problem not supported");
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;
  using ElementScale = ElementC;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor =
      GemmMoeProblemVisitor<ThreadblockShape, kGroupScheduleMode, kThreadCount, kThreadCount, kTransposed>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    int problem_count;
    int threadblock_count;
    int group_size;

    typename EpilogueOutputOp::Params output_op;

    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementScale* weight_scales;
    ElementC* ptr_C;
    ElementC* ptr_D;

    int64_t* total_rows_before_expert;
    int64_t gemm_n;
    int64_t gemm_k;

    // Only used by device-level operator
    GemmCoord* host_problem_sizes;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments()
        : problem_count(0),
          threadblock_count(0),
          ptr_A(nullptr),
          ptr_B(nullptr),
          weight_scales(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          total_rows_before_expert(nullptr),
          gemm_n(0),
          gemm_k(0),
          host_problem_sizes(nullptr) {}

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(int problem_count, int threadblock_count, int group_size, typename EpilogueOutputOp::Params output_op,
              ElementA const* ptr_A, ElementB const* ptr_B, ElementScale const* weight_scales, ElementC const* ptr_C,
              ElementC* ptr_D, int64_t* total_rows_before_expert, int64_t gemm_n, int64_t gemm_k,
              GemmCoord* host_problem_sizes = nullptr)
        : problem_count(problem_count),
          threadblock_count(threadblock_count),
          group_size(group_size),
          output_op(output_op),
          ptr_A(const_cast<ElementA*>(ptr_A)),
          ptr_B(const_cast<ElementB*>(ptr_B)),
          weight_scales(const_cast<ElementScale*>(weight_scales)),
          ptr_C(const_cast<ElementC*>(ptr_C)),
          ptr_D(ptr_D),
          total_rows_before_expert(total_rows_before_expert),
          gemm_n(gemm_n),
          gemm_k(gemm_k),
          host_problem_sizes(nullptr) {
      if (platform::is_same<uint8_t, ElementB>::value || platform::is_same<uint4b_t, ElementB>::value) {
        assert(weight_scales);
      }
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;
    int group_size;

    typename EpilogueOutputOp::Params output_op;

    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementScale* weight_scales;
    ElementC* ptr_C;
    ElementC* ptr_D;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : ptr_A(nullptr), ptr_B(nullptr), weight_scales(nullptr), ptr_C(nullptr), ptr_D(nullptr) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Arguments const& args, void* workspace = nullptr, int tile_count = 0)
        : problem_visitor(args.total_rows_before_expert, args.gemm_n, args.gemm_k, args.problem_count, workspace,
                          tile_count),
          threadblock_count(args.threadblock_count),
          group_size(args.group_size),
          output_op(args.output_op),
          ptr_A(args.ptr_A),
          ptr_B(args.ptr_B),
          weight_scales(args.weight_scales),
          ptr_C(args.ptr_C),
          ptr_D(args.ptr_D) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const& args, void* workspace = nullptr, int tile_count = 0) {
      problem_visitor = typename ProblemVisitor::Params(args.total_rows_before_expert, args.gemm_n, args.gemm_k,
                                                        args.problem_count, workspace, tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      weight_scales = args.weight_scales;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename ProblemVisitor::SharedStorage problem_visitor;
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  MoeFCGemm() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) { return Status::kSuccess; }

  static Status can_implement(Arguments const& args) {
    if (platform::is_same<uint8_t, ElementB>::value || platform::is_same<uint4b_t, ElementB>::value) {
      if (args.weight_scales == nullptr) {
        CUTLASS_TRACE_HOST("MoeFCGemm::can_implement() - weight scales are required for uint8_t and uint4b_t");
        return Status::kInvalid;
      }
    } else if (args.weight_scales != nullptr) {
      CUTLASS_TRACE_HOST(
          "MoeFCGemm::can_implement() - weight scales are ignored for all types except uint8_t and uint4b_t");
      return Status::kInvalid;
    } else if (args.group_size != args.gemm_k) {
      CUTLASS_TRACE_HOST("MoeFCGemm::can_implement() - scale shape should be (1, gemm_n)");
      return Status::kInvalid;
    } else if (static_cast<size_t>(args.gemm_n) < Mma::IteratorB::AccessType::kElements) {
      CUTLASS_TRACE_HOST("MoeFCGemm::can_implement() - gemm_n is smaller than the input alignment");
      return Status::kInvalid;
    }
    return Status::kSuccess;
  }

  static size_t get_extra_workspace_size(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape) {
    return 0;
  }

  CUTLASS_DEVICE
  void run_kernel_(Params const& params, SharedStorage& shared_storage) {
    //
    // These types shadow the type-level definitions and support the ability to implement
    // a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;
    static constexpr int kInterleave = Mma::IteratorB::Shape::kRow / Mma::Shape::kK;
    static_assert(platform::is_same<LayoutB, layout::RowMajor>::value && kInterleave == 1 ||
                      platform::is_same<LayoutB, layout::ColumnMajor>::value && kInterleave >= 1,
                  "B must be row major/col major OR col major interleaved.");

    //
    // Problem visitor.
    //
    ProblemVisitor problem_visitor(params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

    const int64_t gemm_k = params.problem_visitor.gemm_k;
    const int64_t gemm_n = params.problem_visitor.gemm_n;
    int64_t bytes_per_expert_matrix = (gemm_k * gemm_n / 8) * cutlass::sizeof_bits<ElementB>::value;

    // Outer 'persistent' loop to iterate over tiles
    int loop = 0;
    while (problem_visitor.next_tile()) {
      loop++;

      GemmCoord problem_size = problem_visitor.problem_size();
      int32_t problem_idx = problem_visitor.problem_index();
      int32_t cta_idx = int32_t(problem_visitor.threadblock_idx());

      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

      cutlass::gemm::GemmCoord threadblock_offset(static_cast<int>(cta_idx / grid_shape.n()) * Mma::Shape::kM,
                                                  static_cast<int>(cta_idx % grid_shape.n()) * Mma::Shape::kN, 0);

      // Load element pointers. Exchange pointers and strides if working on the transpose
      const int64_t rows_to_jump = problem_idx == 0 ? 0 : params.problem_visitor.last_row_for_problem[problem_idx - 1];
      ElementA* ptr_A = reinterpret_cast<ElementA*>(params.ptr_A) + rows_to_jump * gemm_k;
      typename LayoutA::LongIndex ldm_A = gemm_k;

      char* byte_ptr_B = (reinterpret_cast<char*>(params.ptr_B)) + problem_idx * bytes_per_expert_matrix;
      ElementB* ptr_B = reinterpret_cast<ElementB*>(byte_ptr_B);
      typename LayoutB::LongIndex ldm_B =
          platform::is_same<layout::RowMajor, LayoutB>::value ? gemm_n : gemm_k * kInterleave;

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
          threadblock_offset.m(),
          0,
      };

      cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n() / kInterleave};

      cutlass::MatrixCoord tb_offset_scale{0, threadblock_offset.n()};

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(LayoutA(ldm_A), ptr_A, {problem_size.m(), problem_size.k()}, thread_idx,
                                         tb_offset_A);

      typename Mma::IteratorB iterator_B(LayoutB(ldm_B), ptr_B,
                                         {problem_size.k() * kInterleave, problem_size.n() / kInterleave}, thread_idx,
                                         tb_offset_B);

      typename Mma::FragmentC accumulators;

      accumulators.clear();

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

      int lane_idx = threadIdx.x % 32;

      //
      // Matrix multiply phase
      //

      // Construct thread-scoped matrix multiply
      auto CreateMMA = [&]() {
        if constexpr (use_dq_gemm<Mma>::value)
          return Mma(shared_storage.main_loop, params.group_size, thread_idx, warp_idx, lane_idx);
        else
          return Mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
      };
      Mma mma = CreateMMA();

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();

      // Compute threadblock-scoped matrix multiply-add
      ElementScale* weight_scale_ptr = params.weight_scales + problem_idx * problem_size.n();

      if constexpr (use_dq_gemm<Mma>::value) {
        const MatrixCoord scale_extent = {1, problem_size.n()};
        typename Mma::IteratorScale iterator_scale(Mma::IteratorScale::Layout(scale_extent.column()), weight_scale_ptr,
                                                   scale_extent, thread_idx, tb_offset_scale);

        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_scale, accumulators);
      } else {
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
      }

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      ElementC* ptr_C = reinterpret_cast<ElementC*>(params.ptr_C) + problem_idx * gemm_n;
      ElementC* ptr_D = reinterpret_cast<ElementC*>(params.ptr_D) + rows_to_jump * gemm_n;

      LayoutC layout_C(0);
      LayoutC layout_D(gemm_n);

      typename Epilogue::OutputTileIterator::Params params_C(layout_C);
      typename Epilogue::OutputTileIterator::Params params_D(layout_D);

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(params_C, ptr_C, problem_size.mn(), thread_idx,
                                                       threadblock_offset.mn());

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(params_D, ptr_D, problem_size.mn(), thread_idx,
                                                       threadblock_offset.mn());

      Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op, iterator_D, accumulators, iterator_C);

      // Next tile
      problem_visitor.advance(gridDim.x);
    }
  }

  template <typename CompilationArch>
  CUTLASS_DEVICE void run_kernel(Params const& params, SharedStorage& shared_storage) {
    if constexpr (platform::is_same<KernelArch, CompilationArch>::value) {
      run_kernel_(params, shared_storage);
    } else {
      CUTLASS_NOT_IMPLEMENTED();
    }
  }

  /*
    To improve compilation speed, we do not compile the device operator if the CUDA_ARCH does not correspond
    to the ArchTag of the cutlass kernel operator.
  */
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
    run_kernel<arch::Sm70>(params, shared_storage);
#elif (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
    run_kernel<arch::Sm75>(params, shared_storage);
#elif (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 900)
    run_kernel<arch::Sm80>(params, shared_storage);
#elif (__CUDA_ARCH__ >= 900)
    run_kernel<arch::Sm80>(params,
                           shared_storage);  // Don't compile these for Hopper or later. Use CUTLASS 3.x kernels.
#else
    // static_assert(false,
    //               "Invalid architecture being compiled. Only Volta+ supported in weight-only quantization kernels.");
    ;
#endif
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
