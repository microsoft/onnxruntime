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
 * @file quantb_mma_multistage.h
 * @brief Modified from cutlass/gemm/threadblock/mma_multistage.h.
 * Added the quantized data memory pipeline, dequantization, and feeding
 * to tensor cores. Mainloop pipeline is heavily modified.
 */

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/threadblock/mma_base.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace{

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Utilities for printing layout for the prepacked weights and quantization parameters
///
template<
    /// Data type of the prepacked weights
    typename ElementWeight,
    /// Data type of the quant scales
    typename ElementQScale,
    /// Data type of the quant offsets
    typename ElementQOffset>
struct QuantBLayoutDebug{
  static constexpr bool debug_smem = true;
  static constexpr bool debug_fragment = true;
  ElementWeight* smem_b_ptr_;
  ElementQScale* smem_qscale_ptr_;
  ElementQOffset* smem_qoffset_ptr_;
  int warp_id_;
  int lane_id_;
  int block_id_;

  template<typename Element, int Size>
  CUTLASS_DEVICE
  static void print_fragment(cutlass::Array<Element, Size> const& frag, char label, int block_id, int warp_id, int lane_id){
    static_assert(Size % 4 == 0, "Size must be multiple of 4");
    if constexpr (debug_fragment){
      if (block_id == 1 && warp_id == 0){
        const Element* ptr = reinterpret_cast<const Element*>(&frag);
        for (int i = 0; i < Size/4; i++, ptr+=4){
          if constexpr(std::is_integral<Element>::value){
            printf("T%.2d%c%d, %3d, %3d, %3d, %3d\n",
                   threadIdx.x, label, i,
                   ptr[0], ptr[1], ptr[2], ptr[3]);
          } else {
            printf("T%.2d%c%d, %.3f, %.3f, %.3f, %.3f\n",
                   threadIdx.x, label, i,
                   float(ptr[0]), float(ptr[1]), float(ptr[2]), float(ptr[3]));
          }
        }
      }
    }
  }

  template<typename Element, int Size>
  CUTLASS_DEVICE
  static void print_as_int4(cutlass::Array<Element, Size> const& frag, char label, int block_id, int warp_id, int lane_id){
    constexpr int I8Size = Size * cutlass::sizeof_bits<Element>::value / 8;
    static_assert(I8Size % 2 == 0, "Size must be multiple of 4");
    if constexpr (debug_fragment){
      if (block_id == 1 && warp_id == 0){
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&frag);
        for (int i = 0; i < I8Size/2; i++, ptr+=2){
          printf("T%.2dW%d, %d, %d, %d, %d\n", threadIdx.x, i, ptr[0] & 0x0f, ptr[0] >> 4, ptr[1] & 0x0f, ptr[1] >> 4);
        }
      }
    }
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Dummy type when quant offset is not used, to avoid compilation error,
/// and reduce runtime footprint
///
struct DummyType{
  std::monostate dummy_;
 public:
  DummyType() = default;

  CUTLASS_HOST_DEVICE
  void* data() const {
    return nullptr;
  }

  CUTLASS_HOST_DEVICE
  std::monostate& operator[](int idx) {
    return dummy_;
  }
};

}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class QuantBMmaBase {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  //
  // Dependent types
  //

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,
                              Shape::kN / WarpGemm::kN,
                              Shape::kK / WarpGemm::kK>;

  /// Number of warp-level GEMM oeprations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);

  /// Number of stages
  static int const kStages = Stages;

  static constexpr bool kHasQOffset = !std::is_same<typename Operator::ElementQOffset, std::monostate>::value;

  /// Tensor reference to the A operand
  using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

  /// Tensor reference to the prepacked weights
  using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

  static_assert(kWarpGemmIterations > 1,
                "The pipelined structure requires at least two warp-level "
                "GEMM operations.");

  static_assert((kWarpGemmIterations % 2) == 0,
                "Inner loop iteration must be an even number.");

  // Tensor reference to the quantization scales
  using TensorRefQScale = TensorRef<typename Operator::ElementQScale, typename Operator::SmemLayoutQScale>;
  using TensorRefQOffset = TensorRef<typename Operator::ElementQOffset, typename Operator::SmemLayoutQOffset>;

  // Block size of the quantization (one set of quantization parameters per block of weights)
  using QuantBlocking = typename Operator::QuantBlocking;

  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operand in shared memory
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages +
                                   Policy::SmemPaddingA::kColumn>;

    /// Shape of the prepacked weights in shared memory
    using ShapeB =
        MatrixShape<Shape::kK / 2 * kStages + Policy::SmemPaddingB::kRow,
                    Shape::kN / 2 + Policy::SmemPaddingB::kColumn>;

    /// Shape of the quantization parameter matrix in shared memory
    /// Validation done in mma core class ThreadblockQShape
    using ShapeQScale =
        MatrixShape<(Shape::kK / QuantBlocking::kRow) * kStages,
                    Shape::kN / QuantBlocking::kColumn>;

    using BufTypeQOffset = std::conditional_t<kHasQOffset,
          AlignedBuffer<typename Operator::ElementQOffset, ShapeQScale::kCount>,
          DummyType>;
   public:
    //
    // Data members
    //

    /// Buffer for A operand
    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for prepacked weights
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

    /// Buffer for quantization scales
    AlignedBuffer<typename Operator::ElementQScale, ShapeQScale::kCount> operand_QScale;

    /// Buffer for quantization offsets
    BufTypeQOffset operand_QOffset;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    CUTLASS_HOST_DEVICE
    static typename Operator::SmemLayoutQScale LayoutQMeta() {
      return Operator::SmemLayoutQScale::packed({ShapeQScale::kRow, ShapeQScale::kColumn});
    }

    CUTLASS_HOST_DEVICE
    static typename Operator::SmemLayoutQOffset LayoutQOffset() {
      return Operator::SmemLayoutQOffset::packed({ShapeQScale::kRow, ShapeQScale::kColumn});
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }

    /// Returns a TensorRef to the prepacked weights
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }

    /// Returns a TensorRef to the quantization scales
    CUTLASS_HOST_DEVICE
    TensorRefQScale operand_QScale_ref() {
      return TensorRefQScale{operand_QScale.data(), LayoutQMeta()};
    }

    CUTLASS_HOST_DEVICE
    TensorRefQOffset operand_QOffset_ref() {
      if constexpr (!kHasQOffset){
        return TensorRefQOffset();
      } else {
        return TensorRefQOffset{operand_QOffset.data(), LayoutQOffset()};
      }
    }
  };

 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator::IteratorB warp_tile_iterator_B_;

  /// Iterator to load a warp-scoped tile of quant scales from shared memory
  typename Operator::IteratorQMeta warp_tile_iterator_QScale_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  QuantBMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
      warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx),
      warp_tile_iterator_QScale_(shared_storage.operand_QScale_ref(),
             shared_storage.operand_QOffset_ref(), lane_idx)
  {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Iterators over tiles of quant scales in global memory
    typename IteratorQScale_,
    /// Iterators over tiles of quant scales in shared memory
    typename SmemIteratorQScale_,
    /// Cache operation for quant scales
    cutlass::arch::CacheOperation::Kind CacheOpQScale,
    /// Iterators over tiles of quant scales in global memory
    typename IteratorQOffset_,
    /// Iterators over tiles of quant scales in shared memory
    typename SmemIteratorQOffset_,
    /// Cache operation for quant scales
    cutlass::arch::CacheOperation::Kind CacheOpQOffset,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class QuantBMmaMultistage :
  public QuantBMmaBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = QuantBMmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  using IteratorQScale = IteratorQScale_;
  using IteratorQOffset = IteratorQOffset_;
  using SmemIteratorQScale = SmemIteratorQScale_;
  using SmemIteratorQOffset = SmemIteratorQOffset_;
  using QuantBlocking = typename Base::QuantBlocking;

  static cutlass::arch::CacheOperation::Kind const kCacheOpQScale = CacheOpQScale;
  static cutlass::arch::CacheOperation::Kind const kCacheOpQOffset = CacheOpQOffset;
  static constexpr bool kHasQOffset = Base::kHasQOffset;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of packed weights
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    static int const AsyncCopyIterationsPerStageQScale =
        IteratorQScale::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of quant scale
    static int const kAccessesPerGroupQScale =
        (AsyncCopyIterationsPerStageQScale + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    static int const AsyncCopyIterationsPerStageQOffset =
        IteratorQOffset::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of quant offset
    static int const kAccessesPerGroupQOffset =
        (AsyncCopyIterationsPerStageQOffset + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    // Optional staged-accumulation (e.g., tf32x3 kernels) for improved numerical
    // accuracy, where each mainloop iteration first accumulates into a temporary
    // set of freshly-cleared accumulators, which are subsequently added to the
    // final accumulator set.
    static bool const kStagedAccumulation = arch::UseStagedAccumulation<typename Operator::MathOperator>::value;
  };

 private:


  // Structure encapsulating pipeline state live from one iteration to the next
  struct PipeState {

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

    /// Temporary accumulator to facilitate staged-accumulation
    FragmentC tmp_accum_;

    /// Pair of A fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentA warp_loaded_frag_A_[2];

    /// Pair of B fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentB warp_loaded_frag_B_;
    WarpTransformedFragmentB warp_transformed_frag_B_[2];

    using WarpLoadedFragmentQScale = typename Operator::FragmentQScale;
    WarpLoadedFragmentQScale warp_loaded_frag_QScale_;

    using WarpLoadedFragmentQOffset = typename std::conditional<kHasQOffset,
            typename Operator::FragmentQOffset,
            std::monostate>::type;
    WarpLoadedFragmentQOffset warp_loaded_frag_QOffset_;
  };


 private:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma_;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  /// Iterator to write threadblock-scoped tile of quant meta data to shared memory
  SmemIteratorQScale smem_iterator_QScale_;
  SmemIteratorQOffset smem_iterator_QOffset_;

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;

  /// very small meta data tensor require less threads to load
  bool const should_load_qscale_;
  bool const should_load_qoffset_;

  /// Shared memory pointers for debug dumping
  static constexpr bool debug_layout = false;
  using LayoutDebugType = typename std::conditional<debug_layout,
      QuantBLayoutDebug<typename IteratorB::Element, typename IteratorQScale::Element, typename IteratorQOffset::Element>,
      std::monostate>::type;
  LayoutDebugType layout_debug_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  QuantBMmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
      smem_iterator_QScale_(shared_storage.operand_QScale_ref(), thread_idx),
      smem_iterator_QOffset_(shared_storage.operand_QOffset_ref(), thread_idx),
      should_load_qscale_(thread_idx < IteratorQScale::ThreadMap::kThreads),
      should_load_qoffset_(thread_idx >= IteratorQOffset::kThreadblockSize - IteratorQOffset::ThreadMap::kThreads),
      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension
    if constexpr(debug_layout){
      layout_debug_.smem_b_ptr_ = shared_storage.operand_B_ref().data();
      layout_debug_.smem_qscale_ptr_ = shared_storage.operand_QScale_ref().data();
      if constexpr(kHasQOffset){
        layout_debug_.smem_qoffset_ptr_ = shared_storage.operand_QOffset_ref().data();
      } else {
        layout_debug_.smem_qoffset_ptr_ = nullptr;
      }
      layout_debug_.warp_id_ = warp_idx;
      layout_debug_.lane_id_ = lane_idx;
      layout_debug_.block_id_ = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    }

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_QScale_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  /// Advance shared memory read-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_read_stage()
  {
    ++smem_read_stage_idx_;

    if (smem_read_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      this->warp_tile_iterator_A_.add_tile_offset({0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset({-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      this->warp_tile_iterator_QScale_.add_tile_offset({-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});

      smem_read_stage_idx_ = 0;
    }
  }

  /// Advance global memory read-iterators and shared memory write-iterators to the stage
  CUTLASS_DEVICE
  void advance_smem_write_stage(
    IteratorA &iterator_A,
    IteratorB &iterator_B,
    IteratorQScale &iterator_QScale,
    IteratorQOffset &iterator_QOffset)
  {
    // Advance global iterators
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});
    iterator_QScale.add_tile_offset({1, 0});

    // Advance shared iterators
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});
    smem_iterator_QScale_.add_tile_offset({1, 0});

    if constexpr (kHasQOffset) {
      iterator_QOffset.add_tile_offset({1, 0});
      smem_iterator_QOffset_.add_tile_offset({1, 0});
    }

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_iterator_QScale_.add_tile_offset({-Base::kStages, 0});
      if constexpr (kHasQOffset) {
        smem_iterator_QOffset_.add_tile_offset({-Base::kStages, 0});
      }
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void copy_qscale_tiles(IteratorQScale &iterator_QScale){
    // Quant scale matrix is 1/block_size of the B matrix, for a 64x64 warp tile,
    // it's only 64x64/block_size elements. For blocking size 16 ~ 64, it only
    // takes 4 ~ 16 cp.async instructions to load. One warp has 32 threads, so
    // it should be loaded in less than one cp.async instruction per thread.
    // Even less for quant offset matrix.
    static_assert(Detail::AsyncCopyIterationsPerStageQScale == 1,
                  "Quant scale should be loaded in one shot!");
    static_assert(IteratorQScale::kAccessesPerVector == 1,
                  "Quant scale should 1 access per vector!");

    // Async Copy for quantization scale
    typename IteratorQScale::AccessType *dst_ptr =
        reinterpret_cast<typename IteratorQScale::AccessType *>(
            this->smem_iterator_QScale_.get());

    constexpr int kSrcBytes =
        sizeof_bits<typename IteratorQScale::Element>::value *
            IteratorQScale::ThreadMap::kElementsPerAccess / 8;

    cutlass::arch::cp_async<kSrcBytes, kCacheOpQScale>(
        dst_ptr, iterator_QScale.get(), iterator_QScale.valid());
  }

  CUTLASS_DEVICE
  void copy_qoffset_tiles(IteratorQOffset & iterator_QOffset) {
    static_assert(Detail::AsyncCopyIterationsPerStageQOffset == 1,
                  "Quant offset should be loaded in one shot!");
    static_assert(IteratorQOffset::kAccessesPerVector == 1,
                  "Quant offset should 1 access per vector!");

    if constexpr(kHasQOffset) {
      // Async Copy for quantization offset
      typename IteratorQOffset::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorQOffset::AccessType *>(
              this->smem_iterator_QOffset_.get());

      constexpr int kSrcBytes = sizeof_bits<typename IteratorQOffset::Element>::value *
                                IteratorQOffset::ThreadMap::kElementsPerAccess / 8;

      cutlass::arch::cp_async<kSrcBytes, kCacheOpQOffset>(
            dst_ptr, iterator_QOffset.get(), iterator_QOffset.valid());
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start = 0) {
    auto group_start_A = group_start * Detail::kAccessesPerGroupA;
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();

          cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
              dst_ptr + v, gmem_ptr, iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }

    auto group_start_B = group_start * Detail::kAccessesPerGroupB;
    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B.get();

          cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
              dst_ptr + v, gmem_ptr, iterator_B.valid());

          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A,      ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,      ///< [in|out] iterator over B operand in global memory
    IteratorQScale &iterator_QScale, ///< [in|out] iterator over quant scales in global memory
    IteratorQOffset &iterator_QOffset, ///< [in|out] iterator over quant offsets in global memory
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {

      // Disable global fetching if done with global fetch iterations
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);
      iterator_QScale.clear_mask(gemm_k_iterations == 0 || !should_load_qscale_);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA::Element>::value *
              IteratorA::ThreadMap::kElementsPerAccess /
              IteratorA::kAccessesPerVector / 8;

          int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator_A.get(), iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B.get(), iterator_B.valid());

          ++iterator_B;
        }

        ++this->smem_iterator_B_;
      }

      // Async Copy for quantization scale
      static_assert(Detail::AsyncCopyIterationsPerStageQScale == 1, "Quant scale should be loaded in one shot!");
      static_assert(IteratorQScale::kAccessesPerVector == 1, "Quant scale should 1 access per vector!");

      typename IteratorQScale::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorQScale::AccessType *>(
              this->smem_iterator_QScale_.get());

      constexpr int kSrcBytes =
          sizeof_bits<typename IteratorQScale::Element>::value *
          IteratorQScale::ThreadMap::kElementsPerAccess / 8;

      auto gmem_ptr = iterator_QScale.get();

      cutlass::arch::cp_async<kSrcBytes, kCacheOpQScale>(
          dst_ptr, gmem_ptr, iterator_QScale.valid());

      if constexpr (kHasQOffset) {
        iterator_QOffset.clear_mask(gemm_k_iterations == 0 || !should_load_qoffset_);

        // Async Copy for quantization offset
        static_assert(Detail::AsyncCopyIterationsPerStageQOffset == 1, "Quant offset should be loaded in one shot!");
        static_assert(IteratorQOffset::kAccessesPerVector == 1, "Quant offset should 1 access per vector!");
        typename IteratorQOffset::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorQOffset::AccessType *>(
                this->smem_iterator_QOffset_.get());

        constexpr int kSrcBytes =
            sizeof_bits<typename IteratorQOffset::Element>::value *
                IteratorQOffset::ThreadMap::kElementsPerAccess / 8;

        cutlass::arch::cp_async<kSrcBytes, kCacheOpQOffset>(
            dst_ptr, iterator_QOffset.get(), iterator_QOffset.valid());
      }

      // Move to the next write stage
      advance_smem_write_stage(iterator_A, iterator_B, iterator_QScale, iterator_QOffset);

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }
  }


  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    if constexpr(debug_layout) {
      if (LayoutDebugType::debug_smem && layout_debug_.block_id_ == 1) {
        if (threadIdx.x == 0){
          printf("stage: %d\n", smem_write_stage_idx_);
        }
        cutlass::debug::dump_shmem(layout_debug_.smem_qscale_ptr_, Base::SharedStorage::ShapeQScale::kCount);
        if constexpr(kHasQOffset){
          cutlass::debug::dump_shmem(layout_debug_.smem_qoffset_ptr_, Base::SharedStorage::ShapeQScale::kCount);
        }
      }
    }
  }

  /// Perform a threadblock mainloop iteration of matrix multiply-accumulate
  CUTLASS_DEVICE
  void mac_loop_iter(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC &accum,               ///< [in|out] destination accumulator tile
    IteratorA &iterator_A,          ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,          ///< [in|out] iterator over B operand in global memory
    IteratorQScale &iterator_QScale, ///< [in|out] iterator over quant scales in global memory
    IteratorQOffset &iterator_QOffset, ///< [in|out] iterator over quant offsets in global memory
    int &gemm_k_iterations)         ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Unroll the warp-level MMA tiles of a threadblock's mainloop iteration
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
      // Loading next warp-level tiles from shared memory. This can be skipped on the very
      // last iteration where:
      //   (gemm_k_iterations == (1 - Base::kStages)) && (warp_mma_k == (Base::kWarpGemmIterations - 1))
      // However, evaluating this condition seems more expensive than simply loading the tiles
      this->warp_tile_iterator_QScale_.load(
          pipe_state.warp_loaded_frag_QScale_,
          pipe_state.warp_loaded_frag_QOffset_);
      ++this->warp_tile_iterator_QScale_;

      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_);
      ++this->warp_tile_iterator_B_;

      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      // All warp-tiles issue their share of global->shared fragment copies
      copy_tiles_and_advance(
          iterator_A,
          iterator_B,
          (warp_mma_k + 1) % Base::kWarpGemmIterations);

      if constexpr(debug_layout) {
        if (LayoutDebugType::debug_fragment && layout_debug_.block_id_ == 1 && layout_debug_.warp_id_ == 0 && layout_debug_.lane_id_ == 0){
          printf("LINE %d, warp_tile_B kgroup %d\n", __LINE__, warp_mma_k % Base::kWarpGemmIterations);
        }
        LayoutDebugType::print_as_int4(pipe_state.warp_loaded_frag_B_, 'W', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        LayoutDebugType::print_fragment(Operator::IteratorQScale::debug_expand(pipe_state.warp_loaded_frag_QScale_), 'Q', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        if constexpr(kHasQOffset){
          LayoutDebugType::print_fragment(Operator::IteratorQScale::debug_expand(pipe_state.warp_loaded_frag_QOffset_), 'O', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        }
      }

      warp_mma_.transform(
        pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
        pipe_state.warp_loaded_frag_B_,
        pipe_state.warp_loaded_frag_QScale_,
        pipe_state.warp_loaded_frag_QOffset_);

      if constexpr(debug_layout) {
        LayoutDebugType::print_fragment(pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2], 'B', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
      }

      // Execute the current warp-tile of MMA operations
      if (Detail::kStagedAccumulation) {
        warp_mma_(
          pipe_state.tmp_accum_,
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.tmp_accum_
        );

        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(
          accum,
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          accum
        );
      }

      if (warp_mma_k == 0) {
        copy_qscale_tiles(iterator_QScale);
      }
      if (warp_mma_k == 1) {
        copy_qoffset_tiles(iterator_QOffset);
      }

      // The second-to-last warp-tile also moves to the next global fetch stage
      if (warp_mma_k == Base::kWarpGemmIterations - 2) {
        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_A, iterator_B, iterator_QScale, iterator_QOffset);
        advance_smem_read_stage();

        // Disable global fetching when done with global fetch iterations
        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
        iterator_QScale.clear_mask(gemm_k_iterations == 0 || !should_load_qscale_);
        if constexpr(kHasQOffset){
          iterator_QOffset.clear_mask(gemm_k_iterations == 0 || !should_load_qoffset_);
        }

        // Wait until we have at least one completed global fetch stage
        gmem_wait();
      }

    }
  }

  /// Specialized mainloop iteration of matrix multiply-accumulate, for small M
  CUTLASS_DEVICE
  void mac_loop_iter_small_m(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC &accum,               ///< [in|out] destination accumulator tile
    IteratorA &iterator_A,          ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,          ///< [in|out] iterator over B operand in global memory
    IteratorQScale &iterator_QScale, ///< [in|out] iterator over quant scales in global memory
    IteratorQOffset &iterator_QOffset, ///< [in|out] iterator over quant offsets in global memory
    int &gemm_k_iterations)         ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Unroll the warp-level MMA tiles of a threadblock's mainloop iteration
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
      // In the case of small M, memory latency dominates. We try to move uses far
      // from their definitions to hide latency.
      if constexpr(debug_layout) {
        if (LayoutDebugType::debug_fragment && layout_debug_.block_id_ == 1 && layout_debug_.warp_id_ == 0 && layout_debug_.lane_id_ == 0){
          printf("LINE %d, warp_tile_B kgroup %d\n", __LINE__, warp_mma_k % Base::kWarpGemmIterations);
        }
        LayoutDebugType::print_as_int4(pipe_state.warp_loaded_frag_B_, 'W', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        LayoutDebugType::print_fragment(Operator::IteratorQScale::debug_expand(pipe_state.warp_loaded_frag_QScale_), 'Q', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        if constexpr(kHasQOffset){
          LayoutDebugType::print_fragment(Operator::IteratorQScale::debug_expand(pipe_state.warp_loaded_frag_QOffset_), 'O', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        }
      }

      warp_mma_.transform(
        pipe_state.warp_transformed_frag_B_[(warp_mma_k) % 2],
        pipe_state.warp_loaded_frag_B_,
        pipe_state.warp_loaded_frag_QScale_,
        pipe_state.warp_loaded_frag_QOffset_);

      if constexpr(debug_layout) {
        LayoutDebugType::print_fragment(pipe_state.warp_transformed_frag_B_[(warp_mma_k) % 2], 'B', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
      }

      // Loading next warp-level tiles from shared memory.
      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_);
      ++this->warp_tile_iterator_B_;

      this->warp_tile_iterator_QScale_.load(
          pipe_state.warp_loaded_frag_QScale_,
          pipe_state.warp_loaded_frag_QOffset_);
      ++this->warp_tile_iterator_QScale_;

      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      // All warp-tiles issue their share of global->shared fragment copies
      copy_tiles_and_advance(
          iterator_A,
          iterator_B,
          (warp_mma_k + 1) % Base::kWarpGemmIterations);

      // Execute the current warp-tile of MMA operations
      if (Detail::kStagedAccumulation) {
        warp_mma_(
          pipe_state.tmp_accum_,
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.tmp_accum_
        );

        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(
          accum,
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          accum
        );
      }

      // The second-to-last warp-tile also moves to the next global fetch stage
      if (warp_mma_k == Base::kWarpGemmIterations - 2) {
        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_A, iterator_B, iterator_QScale, iterator_QOffset);
        advance_smem_read_stage();

        // Disable global fetching when done with global fetch iterations
        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
        iterator_QScale.clear_mask(gemm_k_iterations == 0 || !should_load_qscale_);
        if constexpr(kHasQOffset){
          iterator_QOffset.clear_mask(gemm_k_iterations == 0 || !should_load_qoffset_);
        }

        copy_qscale_tiles(iterator_QScale);
        copy_qoffset_tiles(iterator_QOffset);

        // Wait until we have at least one completed global fetch stage
        gmem_wait();
      }

    }
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
      int gemm_k_iterations,        ///< number of threadblock mainloop iterations
      FragmentC &accum,             ///< [in|out] accumulator tile
      IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
      IteratorB &iterator_B,        ///< [in|out] iterator over B operand in global memory
      IteratorQScale &iterator_QScale, ///< [in|out] iterator over QScale operand in global memory
      IteratorQOffset &iterator_QOffset) ///< [in|out] iterator over QOffset operand in global memory
  {
    PipeState pipe_state;

    // Disable global fetching if done with global fetch iterations
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);
    iterator_QScale.clear_mask(gemm_k_iterations == 0 || !should_load_qscale_);
    if constexpr(kHasQOffset) {
      iterator_QOffset.clear_mask(gemm_k_iterations == 0 || !should_load_qoffset_);
    }

    // Load first warp-tile's B fragment from shared memory
    this->warp_tile_iterator_QScale_.load(
        pipe_state.warp_loaded_frag_QScale_,
        pipe_state.warp_loaded_frag_QOffset_);
    ++this->warp_tile_iterator_QScale_;

    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_);
    ++this->warp_tile_iterator_B_;

    // Load first warp-tile's A fragment from shared memory
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    copy_tiles_and_advance(iterator_A, iterator_B, 0);

    if constexpr(Shape::kM > 32) {
      // the case of bigger m
      if constexpr(debug_layout) {
        if (LayoutDebugType::debug_fragment && layout_debug_.block_id_ == 1 && layout_debug_.warp_id_ == 0 && layout_debug_.lane_id_ == 0){
          printf("LINE %d, warp_tile_B kgroup %d\n", __LINE__, 0);
        }
        LayoutDebugType::print_as_int4(pipe_state.warp_loaded_frag_B_, 'W', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        LayoutDebugType::print_fragment(Operator::IteratorQScale::debug_expand(pipe_state.warp_loaded_frag_QScale_), 'Q', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        if constexpr(kHasQOffset){
          LayoutDebugType::print_fragment(Operator::IteratorQScale::debug_expand(pipe_state.warp_loaded_frag_QOffset_), 'O', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
        }
      }

      warp_mma_.transform(
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_B_,
        pipe_state.warp_loaded_frag_QScale_,
        pipe_state.warp_loaded_frag_QOffset_);

      if constexpr(debug_layout) {
        LayoutDebugType::print_fragment(pipe_state.warp_transformed_frag_B_[0], 'B', layout_debug_.block_id_, layout_debug_.warp_id_, layout_debug_.lane_id_);
      }
    } else {
      // the case of small m
      copy_qscale_tiles(iterator_QScale);
      copy_qoffset_tiles(iterator_QOffset);
    }

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }

    // Mainloop
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      if constexpr(Shape::kM > 32) {
        mac_loop_iter(
          pipe_state,
          accum,
          iterator_A,
          iterator_B,
          iterator_QScale,
          iterator_QOffset,
          gemm_k_iterations);
      } else {
        mac_loop_iter_small_m(
          pipe_state,
          accum,
          iterator_A,
          iterator_B,
          iterator_QScale,
          iterator_QOffset,
          gemm_k_iterations);
      }
    }

    if (Detail::kStagedAccumulation) {
      plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

  }


  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< iterator over quant scales in global memory
      IteratorQScale iterator_QScale,
      ///< Iterator over quant offsets in global memory
      IteratorQOffset iterator_QOffset,
      ///< initial value of accumulator
      FragmentC const &src_accum) {

    // Prologue (start fetching iterations of global fragments into shared memory)
    prologue(iterator_A, iterator_B, iterator_QScale, iterator_QOffset, gemm_k_iterations);

    // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // Initialize destination accumulators with source accumulators
    accum = src_accum;

    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B, iterator_QScale, iterator_QOffset);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
