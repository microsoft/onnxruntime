/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * @file quantb_meta_mma_tensor_op_tile_iterator.h
 * @brief Templates for loading quantization meta data for operand B
 *        from shared memory to fragments. This is meant to be used in
 *        lock step with the operand B tile iterator. Containing logic
 *        to figure out the operand B layout in the tensor core,
 *        and deliver each meta data element to its corresponding
 *        operand B element for dequantization.
 */

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace{

struct b32_pair{
  uint32_t a;
  uint32_t b;
};

struct fp16_quad{
  cutlass::half_t a;
  cutlass::half_t b;
  cutlass::half_t c;
  cutlass::half_t d;
};

struct b16_quad{
  int16_t a;
  int16_t b;
  int16_t c;
  int16_t d;
};

union b64 {
  uint64_t single;
  b32_pair pair;
  b16_quad quard;
  fp16_quad fp16_quad;
};

static_assert(sizeof(b64) == 8, "b64 should be 64 bits");

/// Convert packed 4b weights into fp16(weight + 16)
/// Current bit hacking only supports fp16, need to add bf16 later.
///
template<int Size>
CUTLASS_DEVICE
void weights2Half(cutlass::Array<uint8_t,Size/2> const &weights,
                 cutlass::Array<cutlass::half_t, Size>& dest)
{
  static_assert(Size % 8 == 0, "Weights should have been prepacked by 2x2 tiles, 2 weights per tile.");
  uint32_t* dest_pair = reinterpret_cast<uint32_t*>(dest.data());
  const uint32_t* w_oct = reinterpret_cast<const uint32_t*>(weights.data());

  CUTLASS_PRAGMA_UNROLL
  for (int oct_idx = 0; oct_idx < Size/8; oct_idx++, w_oct++, dest_pair += 4){
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

    // static_cast<cutlass::half_t>(16 + weight)
    // 4b weights are prepacked into [0, 2, 4, 6, 1, 3, 5, 7], so that adjacent weights
    // are in different 16b half words, making it easier to convert to fp16.
    asm volatile(
        "{\n\t"
        "  shl.b32       %0, %4, 6;\n"
        "  shl.b32       %1, %4, 2;\n"
        "  shr.u32       %2, %4, 2;\n"
        "  shr.u32       %3, %4, 6;\n"
        "  lop3.b32      %0, %0, 0x03c003c0, 0x4c004c00, 0xea;\n" // a & 0x03c0 | 0x4c00
        "  lop3.b32      %1, %1, 0x03c003c0, 0x4c004c00, 0xea;\n"
        "  lop3.b32      %2, %2, 0x03c003c0, 0x4c004c00, 0xea;\n"
        "  lop3.b32      %3, %3, 0x03c003c0, 0x4c004c00, 0xea;\n"
        "}\n"
        : "=r"(dest_pair[0]), "=r"(dest_pair[1]),
          "=r"(dest_pair[2]), "=r"(dest_pair[3])
        : "r"(*w_oct));
#else
    assert(0);
#endif
  }

}

} // namespace

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

// Traits to describe the layout of quantization meta data layout in a MMA fragment
// Since operand B is quantized on a per block basis, it's one meta data per block.

template <
  /// Shape of the operand B matrix to load in a warp (concept: MatrixShape<kK, kN>)
  typename WarpShapeB_,
  /// Block dimensions of the blockwise quantization. So the actual meta data
  /// warp shape is WarpShapeB_ / BlockingShape_
  typename BlockingShape_,
  /// Underlying matrix multiply operator (concept: arch::Mma)
  typename ArchMmaOperator_,
  /// Number of threads participating in one matrix operation
  int Threads>
class QuantBMetaMmaTile{
public:

  using WarpShapeB = WarpShapeB_;
  using BlockingShape = BlockingShape_;
  using ArchMmaOperator = ArchMmaOperator_;

  static_assert(Threads == 32, "This iterator should work in a warp only.");

  /// Shape of the curresponding operand B tile iterator <instruction_k, warp_n>
  using TileShapeB = MatrixShape<ArchMmaOperator::Shape::kK, WarpShapeB::kColumn>;

  // Tensor core operand B layout is a column major 4x8 tile, divided
  // into 32 threads (T0 ~ T31) as shown below. Each element of the tile is 32b,
  // so for fp16 it becomes 8 x 8, and int8 it becomes 16 x 8.
  //  T0 |  T4 |  T8 | T12 | T16 | T20 | T24 | T28
  //  T1 |  T5 |  T9 | T13 | T17 | T21 | T25 | T29
  //  T2 |  T6 | T10 | T14 | T18 | T22 | T26 | T30
  //  T3 |  T7 | T11 | T15 | T19 | T23 | T27 | T31
  using CoreTile = layout::PitchLinearShape<4, 8>;

  /// Each thread holds a 32b fragment per tile: for half precision, it's 2 elements, 4 elements for int8
  static int const kNumBsPerCoreTileFragement = 32 / sizeof_bits<typename ArchMmaOperator::ElementB>::value;

  /// Each mma instruction can process either 1 or 2 tensor core operand B tiles (stacked on the k dimension)
  static int const kBTilesPerMma =
      sizeof_bits<typename ArchMmaOperator::ElementB>::value * ArchMmaOperator::FragmentB::kElements / 32;
  static_assert(kBTilesPerMma == 1 || kBTilesPerMma == 2, "Only support 1 or 2 operand B tiles per mma.");

  /// Each operand B tile iterator load covers a number of mma instructions
  static int const kMmaIterationsB = WarpShapeB::kColumn / ArchMmaOperator::Shape::kN;

  /// Number of B elements a fragment of meta data should cover
  static int const kExpandedSize = kNumBsPerCoreTileFragement * kBTilesPerMma * kMmaIterationsB;

  // Now we figure out how many meta data elements to load for each TileShapeB

  /// Number of meta elements per CoreTile.
  static int const kCoreTileFragementSize = (kNumBsPerCoreTileFragement + BlockingShape::kRow - 1) / BlockingShape::kRow;

  /// Number of core tiles per mma instruction, different from kBTilesPerMma when blocking size on K dimension
  /// exceeds the tile depth, so two tiles share the same meta data
  static int const kTilesPerMma = ((kBTilesPerMma == 2) &&
                                  (BlockingShape::kRow <= kNumBsPerCoreTileFragement * CoreTile::kContiguous))
                                  ? 2 : 1;

  /// stride to reach the meta data for the next CoreTile on the K dimension
  static int const kKTileStride = (kNumBsPerCoreTileFragement * CoreTile::kContiguous + BlockingShape::kRow - 1) / BlockingShape::kRow;

  /// Stride on N dimension should be the tile width, shrunk by blocking size on this dimension.
  static int const kNStride = (CoreTile::kStrided + BlockingShape::kColumn - 1) / BlockingShape::kColumn;

  /// On N dimension, how many tiles share the same meta data
  static int const kNRepeats = (BlockingShape::kColumn + CoreTile::kStrided - 1) / CoreTile::kStrided;

  /// Each fragment should cover kMmaIterationsB number of mma intructions on the N dimension.
  /// When blocking size on this dimension exceeds the tile width, multiple iterations
  /// would share the same data.
  static int const kMmaIterations = (kMmaIterationsB + kNRepeats - 1) / kNRepeats;

  static int const kFragementSize = kCoreTileFragementSize * kTilesPerMma * kMmaIterations;

  CUTLASS_DEVICE
  static MatrixCoord lane_position(int lane_id) {
    if constexpr(kNumBsPerCoreTileFragement == 2
                 && kBTilesPerMma == 2
                 && BlockingShape::kRow == 1){
      // Optimize for a special case of:
      //    16b gemm (kNumBsPerCoreTileFragement == 2)
      //    2 B operand tiles per mma (kBTilesPerMma == 2)
      //    (1,n) quantization blocking
      // The scale and offset tensors are prepacked to reduce the number of load instructions.
      return make_Coord((lane_id % CoreTile::kContiguous) * 4,
         lane_id / CoreTile::kContiguous);
    } else {
      return make_Coord((lane_id % CoreTile::kContiguous) * kNumBsPerCoreTileFragement,
         lane_id / CoreTile::kContiguous);
    }
  }
};


////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is to load quantization meta data for operand B from
/// shared memory to fragments (hopefully allocated to registers by compilers).
/// Examples of meta data include scale or offsets. The operand B matrix is
/// quantized on a per block basis, meaning one element of meta data per block.
///
/// This is meant to be used in lock step with the operand B tile iterator.
/// So all parameters are logical positions in the operand B tiles.
/// The goal here is to deliver each meta data element to its corresponding
/// operand B element for dequantization. As a result, we need to figure
/// out the operand B layout in the tensor core.
///
template <
  /// Shape of the operand B matrix to load in a warp (concept: MatrixShape<kK, kN>)
  typename WarpShapeB_,
  /// Block dimensions of the blockwise quantization. So the actual meta data
  /// warp shape is WarpShapeB_ / BlockingShape_
  typename BlockingShape_,
  /// Data type of the quant scales
  typename ElementScale_,
  /// Layout of the quant scales
  typename LayoutScale_,
  /// Data type of quant offsets
  typename ElementOffset_,
  /// Layout of quant offsets
  typename LayoutOffset_,
  /// Underlying matrix multiply operator (concept: arch::Mma)
  typename ArchMmaOperator_,
  /// Number of threads participating in one matrix operation
  int Threads,
  /// Number of partitions along K dimension
  int PartitionsK_ = 1>
class QuantBMetaMmaTensorOpTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for column major layout

template <
  /// Shape of the operand B matrix to load in a warp (concept: MatrixShape<kK, kN>)
  typename WarpShapeB_,
  /// Block dimensions of the blockwise quantization. So the actual meta data
  /// warp shape is WarpShapeB_ / BlockingShape_
  typename BlockingShape_,
  /// Data type of the meta data elements
  typename ElementScale_,
  /// Data type of quant offsets
  typename ElementOffset_,
  /// Underlying matrix multiply operator (concept: arch::Mma)
  typename ArchMmaOperator_,
  /// Number of threads participating in one matrix operation
  int Threads>
class QuantBMetaMmaTensorOpTileIterator<WarpShapeB_, BlockingShape_,
    ElementScale_, cutlass::layout::ColumnMajor,
    ElementOffset_, cutlass::layout::ColumnMajor,
    ArchMmaOperator_, Threads, 1>{
public:

  using WarpShapeB = WarpShapeB_;
  using BlockingShape = BlockingShape_;
  using ElementScale = ElementScale_;
  using Layout = cutlass::layout::ColumnMajor;
  using ElementOffset = ElementOffset_;
  using ArchMmaOperator = ArchMmaOperator_;

  static constexpr bool kHasOffset = !(std::is_same<ElementOffset, std::monostate>::value);

  static_assert(BlockingShape::kRow == 1 && BlockingShape::kColumn > 1,
          "Only support row blocking for column major layout");

  using MetaTile = QuantBMetaMmaTile<WarpShapeB, BlockingShape, ArchMmaOperator, Threads>;

  /// Number of MMA instructions for this tile
  static constexpr int kMmaIterationsB = MetaTile::kMmaIterationsB;

  /// Number of B elements per mma tile fragment (32b), 2 for half precision, 4 for int8
  static constexpr int kNumBsPerCoreTileFragement = MetaTile::kNumBsPerCoreTileFragement;

  /// Each mma instruction can process either 1 or 2 operand B tiles (stacked on the k dimension)
  static constexpr int kBTilesPerMma = MetaTile::kBTilesPerMma;

  /// Number of B elements a fragment of meta data should cover
  static constexpr int kExpandedSize = MetaTile::kExpandedSize;

  /// Number of meta elements per core tile fragment
  static constexpr int kCoreTileFragementSize = MetaTile::kCoreTileFragementSize;

  /// stride for reaching the next core tile (if there is one) on the K dimension
  static constexpr int kKTileStride = MetaTile::kKTileStride;

  /// do we need to load meta data for the next core tile on the K dimension?
  static constexpr int kTilesPerMma = MetaTile::kTilesPerMma;

  static constexpr int kNStride = MetaTile::kNStride;
  static constexpr int kNRepeats = MetaTile::kNRepeats;
  static constexpr int kMmaIterations = MetaTile::kMmaIterations;

  using TensorRefScale = TensorRef<ElementScale, Layout>;
  using TensorRefOffset = TensorRef<ElementOffset, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using FragmentScale = Array<ElementScale, MetaTile::kFragementSize>;
  using FragmentOffset = typename std::conditional<kHasOffset,
          Array<ElementOffset, MetaTile::kFragementSize>,
          std::monostate>::type;

  using AccessTypeScale = Array<ElementScale, kCoreTileFragementSize>;
  using AccessTypeOffset = Array<ElementOffset, kCoreTileFragementSize>;

private:

  ElementScale *pointer_;
  Layout layout_;

  ElementOffset *pointer_offset_;
  Layout layout_offset_;

  TensorCoord lane_position_;

public:

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator() { }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator(
    TensorRefScale const &ref,
    TensorRefOffset const &ref_offset,
    int lane_idx
  ):
    pointer_(ref.data()),
    layout_(ref.layout()),
    pointer_offset_(ref_offset.data()),
    layout_offset_(ref_offset.layout()),
    lane_position_(MetaTile::lane_position(lane_idx)){}

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(FragmentScale &frag, FragmentOffset &frag_offset) {
    if constexpr(kNumBsPerCoreTileFragement == 2
                 && kBTilesPerMma == 2){
      // Optimize for a special case of:
      //    16b gemm (kNumBsPerCoreTileFragement == 2)
      //    2 B operand tiles per mma (kBTilesPerMma == 2)
      //    (1,n) quantization blocking (BlockingShape::kRow == 1)
      // The scale and offset tensors are prepacked to reduce the number of load instructions needed
      const int row = lane_position_.row();
      const int column = lane_position_.column() / BlockingShape::kColumn;

      Array<ElementScale, 4> *dst_ptr = reinterpret_cast<Array<ElementScale, 4>*>(frag.data());
      CUTLASS_PRAGMA_UNROLL
      for (int n_idx = 0, c = column; n_idx < kMmaIterations; n_idx++, c += kNStride){
        Array<ElementScale, 4> *src_ptr = reinterpret_cast<Array<ElementScale, 4>*>(pointer_ + layout_({row, c}));
        *dst_ptr = *src_ptr;
        dst_ptr++;
      }

      if constexpr(kHasOffset){
        Array<ElementOffset, 4> *dst_ptr_offset = reinterpret_cast<Array<ElementOffset, 4>*>(frag_offset.data());
        CUTLASS_PRAGMA_UNROLL
        for (int n_idx = 0, c = column; n_idx < kMmaIterations; n_idx++, c += kNStride){
          Array<ElementOffset, 4> *src_ptr_offset = reinterpret_cast<Array<ElementOffset, 4>*>(pointer_offset_ + layout_offset_({row, c}));
          *dst_ptr_offset = *src_ptr_offset;
          dst_ptr_offset++;
        }
      }

    } else {
      // Other cases, offsets and scales are not prepacked.

      const int row = lane_position_.row() / BlockingShape::kRow;
      const int column = lane_position_.column() / BlockingShape::kColumn;

      AccessTypeScale* dst_ptr = reinterpret_cast<AccessTypeScale*>(frag.data());
      CUTLASS_PRAGMA_UNROLL
      for (int n_idx = 0, c = column; n_idx < kMmaIterations; n_idx++, c += kNStride){
        CUTLASS_PRAGMA_UNROLL
        for (int mma_tile_idx = 0, r = row; mma_tile_idx < kTilesPerMma; mma_tile_idx++, r += kKTileStride){
          AccessTypeScale* src_ptr = reinterpret_cast<AccessTypeScale*>(pointer_ + layout_({r, c}));
          *dst_ptr = *src_ptr;
          dst_ptr++;
        }
      }

      if constexpr(kHasOffset){
        AccessTypeOffset* dst_ptr = reinterpret_cast<AccessTypeOffset*>(frag_offset.data());
        CUTLASS_PRAGMA_UNROLL
        for (int n_idx = 0, c = column; n_idx < kMmaIterations; n_idx++, c += kNStride){
          CUTLASS_PRAGMA_UNROLL
          for (int mma_tile_idx = 0, r = row; mma_tile_idx < kTilesPerMma; mma_tile_idx++, r += kKTileStride){
            AccessTypeOffset* src_ptr = reinterpret_cast<AccessTypeOffset*>(pointer_offset_ + layout_offset_({r, c}));
            *dst_ptr = *src_ptr;
            dst_ptr++;
          }
        }
      }
    }
  }

  template <typename ElementT>
  CUTLASS_HOST_DEVICE
  static Array<ElementT, kExpandedSize> debug_expand(Array<ElementT, MetaTile::kFragementSize> const &frag){
    Array<ElementT, kExpandedSize> ret;
    int out_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      int n_idx = n_out / kNRepeats;
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        int mma_tile_idx = mma_tile_out_idx / (kBTilesPerMma / kTilesPerMma);
        CUTLASS_PRAGMA_UNROLL
        for (int elem_out_idx = 0; elem_out_idx < kNumBsPerCoreTileFragement; elem_out_idx++){
          int elem_idx = elem_out_idx / BlockingShape::kRow;
          int idx = elem_idx + mma_tile_idx * kCoreTileFragementSize + n_idx * kCoreTileFragementSize * kTilesPerMma;
          ret[out_idx] = frag[idx];
          out_idx++;
        }
      }
    }
    return ret;
  }

  CUTLASS_HOST_DEVICE
  static void dequant(FragmentScale const &scales,
                      FragmentOffset const &offsets,
                      Array<uint8_t,kExpandedSize/2> const &weights,
                      Array<ElementScale, kExpandedSize>& dest){
    static_assert(kNumBsPerCoreTileFragement == 2, "Only for 16b gemm.");
    static_assert(kExpandedSize % 8 == 0, "Weights should have been prepacked by 2x2 tiles, 2 weights per tile.");

    // First convert 4b weight into fp16(weight + 16)
    weights2Half(weights, dest);

    if constexpr(kBTilesPerMma == 2){
      // Optimize for a special case of:
      //    2 B operand tiles per mma (kBTilesPerMma == 2)
      //    (1,n) quantization blocking (BlockingShape::kRow == 1)

      uint32_t* dest_pair = reinterpret_cast<uint32_t*>(dest.data());
      const b64* scales_ptr = reinterpret_cast<const b64*>(scales.data());
      const ElementOffset* offsets_ptr = nullptr;
      if constexpr(kHasOffset) { offsets_ptr = offsets.data(); }

      CUTLASS_PRAGMA_UNROLL
      for (int n_idx = 0; n_idx < kMmaIterations; n_idx++){
        // dequantize: d = scale * (weight - offset)
        // to use FMA, d = scale * weight + (scale * (-offset))

        b64 offsets;
        if constexpr(kHasOffset){
          const uint32_t* p = reinterpret_cast<const uint32_t*>(offsets_ptr);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
          asm volatile(
              "{\n\t"
              "  .reg  .b32    rb0, rb1;\n"      // b32 regs for fp16x2 mul operands

              // static_cast<cutlass::half_t>(-16 - offset)
              // input [d, b, c, a],
              "  shl.b32       rb0, %4, 6;\n"     // rb0 = [x, b, x, a] << 6
              "  shr.u32       rb1, %4, 2;\n"     // rb1 = [x, d, x, c] << 6
              "  lop3.b32      rb0, rb0, 0x03c003c0, 0xcc00cc00, 0xea;\n" // a & 0x03c0 | 0xcc00
              "  lop3.b32      rb1, rb1, 0x03c003c0, 0xcc00cc00, 0xea;\n"
              "  mul.rn.f16x2  %0, %2, rb0;\n"    // offset = scale * (-16 - offset)
              "  mul.rn.f16x2  %1, %3, rb1;\n"
              "}\n"
              : "=r"(offsets.pair.a), "=r"(offsets.pair.b)
              : "r"(scales_ptr->pair.a), "r"(scales_ptr->pair.b),
                "r"(p[0]));
#else
          assert(0);
#endif

          offsets_ptr += 4;
        } else {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
          asm volatile(
              "{\n\t"
              "  .reg  .b32    rb0;\n"
              "  mov.u32       rb0, 0xce00ce00;\n"
              "  mul.rn.f16x2  %0, %2, rb0;\n"    // offset = scale * (-16 - 8)
              "  mul.rn.f16x2  %1, %3, rb0;\n"
              "}\n"
              : "=r"(offsets.pair.a), "=r"(offsets.pair.b)
              : "r"(scales_ptr->pair.a), "r"(scales_ptr->pair.b));
#else
          offsets.fp16_quad.a = scales_ptr->fp16_quad.a * static_cast<cutlass::half_t>(-16-8);
          offsets.fp16_quad.b = scales_ptr->fp16_quad.b * static_cast<cutlass::half_t>(-16-8);
          offsets.fp16_quad.c = scales_ptr->fp16_quad.c * static_cast<cutlass::half_t>(-16-8);
          offsets.fp16_quad.d = scales_ptr->fp16_quad.d * static_cast<cutlass::half_t>(-16-8);
#endif
        }

        CUTLASS_PRAGMA_UNROLL
        for (int n_r = 0; n_r < kNRepeats; n_r++){
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
          asm volatile(
              "{\n\t"
              "  fma.rn.f16x2  %0, %2, %0, %4;\n" // dest = scale * (16 + weight) +  (scale * (-16 - offset))
              "  fma.rn.f16x2  %1, %3, %1, %5;\n"
              "}\n"
              : "+r"(dest_pair[0]), "+r"(dest_pair[1])
              : "r"(scales_ptr->pair.a), "r"(scales_ptr->pair.b),
                "r"(offsets.pair.a), "r"(offsets.pair.b));
#else
          assert(0);
#endif
          dest_pair += 2;
        }
        scales_ptr++;
      }

    } else {
      // unoptiomized path for other cases, very slow
      int out_idx = 0;
      ElementScale offset;
      CUTLASS_PRAGMA_UNROLL
      for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
        int n_idx = n_out / kNRepeats;
        CUTLASS_PRAGMA_UNROLL
        for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
          int mma_tile_idx = mma_tile_out_idx / (kBTilesPerMma / kTilesPerMma);
          CUTLASS_PRAGMA_UNROLL
          for (int elem_out_idx = 0; elem_out_idx < kNumBsPerCoreTileFragement; elem_out_idx++){
            int elem_idx = elem_out_idx / BlockingShape::kRow;
            int idx = elem_idx + mma_tile_idx * kCoreTileFragementSize + n_idx * kCoreTileFragementSize * kTilesPerMma;
            ElementScale s = scales[idx];
            if constexpr(kHasOffset){
              offset = s * static_cast<ElementScale>(-16 - int(offsets[idx]));
            } else {
              offset = s * static_cast<ElementScale>(-16-8);
            }
            dest[out_idx] = s * dest[out_idx] + offset;
            out_idx++;
          }
        }
      }

    }

  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  QuantBMetaMmaTensorOpTileIterator &operator++() {
    // This is for operand B, so advance on the K dimension
    lane_position_ += make_Coord(MetaTile::TileShapeB::kRow, 0);
    return *this;
  }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int rows = tile_offset.row() * MetaTile::TileShapeB::kRow;
    int columns = tile_offset.column() * MetaTile::TileShapeB::kColumn;
    lane_position_ += TensorCoord(rows, columns);
    return *this;
  }

};


////////////////////////////////////////////////////////////////////////////////

/// Specialization for row major layout

template <
  /// Shape of the operand B matrix to load in a warp (concept: MatrixShape<kK, kN>)
  typename WarpShapeB_,
  /// Block dimensions of the blockwise quantization. So the actual meta data
  /// warp shape is WarpShapeB_ / BlockingShape_
  typename BlockingShape_,
  /// Data type of the meta data elements
  typename ElementScale_,
  /// Data type of quant offsets
  typename ElementOffset_,
  /// Underlying matrix multiply operator (concept: arch::Mma)
  typename ArchMmaOperator_,
  /// Number of threads participating in one matrix operation
  int Threads>
class QuantBMetaMmaTensorOpTileIterator<WarpShapeB_, BlockingShape_,
    ElementScale_, cutlass::layout::RowMajor,
    ElementOffset_, cutlass::layout::RowMajor,
    ArchMmaOperator_, Threads, 1>{
public:

  using WarpShapeB = WarpShapeB_;
  using BlockingShape = BlockingShape_;
  using ElementScale = ElementScale_;
  using ElementOffset = ElementOffset_;
  using Layout = cutlass::layout::RowMajor;
  using ArchMmaOperator = ArchMmaOperator_;

  static constexpr bool kHasOffset = !(std::is_same<ElementOffset, std::monostate>::value);

  static_assert(BlockingShape::kColumn == 1 && BlockingShape::kRow > 1,
          "Only support column blocking for row major layout");

  using MetaTile = QuantBMetaMmaTile<WarpShapeB, BlockingShape, ArchMmaOperator, Threads>;

  /// Number of MMA instructions for this tile
  static constexpr int kMmaIterationsB = MetaTile::kMmaIterationsB;

  /// Number of B elements per mma tile fragment (32b), 2 for half precision, 4 for int8
  static constexpr int kNumBsPerCoreTileFragement = MetaTile::kNumBsPerCoreTileFragement;

  /// Each mma instruction can process either 1 or 2 operand B tiles (stacked on the k dimension)
  static constexpr int kBTilesPerMma = MetaTile::kBTilesPerMma;

  /// Number of B elements a fragment of meta data should cover
  static constexpr int kExpandedSize = MetaTile::kExpandedSize;

  /// Number of meta elements per core tile fragment
  static constexpr int kCoreTileFragementSize = MetaTile::kCoreTileFragementSize;

  /// stride for reaching the next core tile (if there is one) on the K dimension
  static constexpr int kKTileStride = MetaTile::kKTileStride;

  /// do we need to load meta data for the next core tile on the K dimension?
  static constexpr int kTilesPerMma = MetaTile::kTilesPerMma;

  static constexpr int kNStride = MetaTile::kNStride;
  static constexpr int kNRepeats = MetaTile::kNRepeats;
  static constexpr int kMmaIterations = MetaTile::kMmaIterations;

  using TensorRefScale = TensorRef<ElementScale, Layout>;
  using TensorRefOffset = TensorRef<ElementOffset, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using FragmentScale = Array<ElementScale, MetaTile::kFragementSize>;
  using FragmentOffset = typename std::conditional<kHasOffset,
          Array<ElementOffset, MetaTile::kFragementSize>,
          std::monostate>::type;

private:

  ElementScale *pointer_;
  Layout layout_;

  ElementOffset *pointer_offset_;
  Layout layout_offset_;

  TensorCoord lane_position_;

public:

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator() { }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator(
    TensorRefScale const &ref,
    TensorRefOffset const &ref_offset,
    int lane_idx
  ):
    pointer_(ref.data()),
    layout_(ref.layout()),
    pointer_offset_(ref_offset.data()),
    layout_offset_(ref_offset.layout()),
    lane_position_(MetaTile::lane_position(lane_idx))
     {}

  /// Loads a fragment
  CUTLASS_HOST_DEVICE
  void load(FragmentScale &frag, FragmentOffset &frag_offset) {
    const int row = lane_position_.row() / BlockingShape::kRow;
    const int column = lane_position_.column() / BlockingShape::kColumn;
    static_assert(kTilesPerMma * kCoreTileFragementSize == 1, "Only support one meta data per core tile");

    ElementScale* src_ptr = pointer_ + layout_({row, column});
    ElementScale* dst_ptr = frag.data();
    CUTLASS_PRAGMA_UNROLL
    for (int n_idx = 0; n_idx < kMmaIterations; n_idx++){
      dst_ptr[n_idx] = src_ptr[n_idx * kNStride];
    }

    if constexpr(kHasOffset){
      ElementOffset* src_ptr_offset = pointer_offset_ + layout_offset_({row, column});
      ElementOffset* dst_ptr_offset = frag_offset.data();
      CUTLASS_PRAGMA_UNROLL
      for (int n_idx = 0; n_idx < kMmaIterations; n_idx++){
        dst_ptr_offset[n_idx] = src_ptr_offset[n_idx * kNStride];
      }
    }
  }

  template <typename ElementT>
  CUTLASS_HOST_DEVICE
  static Array<ElementT, kExpandedSize> debug_expand(Array<ElementT, MetaTile::kFragementSize> const &frag){
    Array<ElementT, kExpandedSize> ret;

    int out_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      int n_idx = n_out / kNRepeats;
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        int mma_tile_idx = mma_tile_out_idx / (kBTilesPerMma / kTilesPerMma);
        CUTLASS_PRAGMA_UNROLL
        for (int elem_out_idx = 0; elem_out_idx < kNumBsPerCoreTileFragement; elem_out_idx++){
          int elem_idx = elem_out_idx / BlockingShape::kRow;
          int col = elem_idx + mma_tile_idx * kCoreTileFragementSize;
          int idx = col * kMmaIterations + n_idx;
          ret[out_idx] = frag[idx];
          out_idx++;
        }
      }
    }
    return ret;
  }

  CUTLASS_HOST_DEVICE
  static void dequant(FragmentScale const &scales,
                      FragmentOffset const &offsets,
                      Array<uint8_t,kExpandedSize/2> const &weights,
                      Array<ElementScale, kExpandedSize>& dest){
    static_assert(kNRepeats == 1, "This is implied by BlockingShape::kColumn == 1");
    static_assert(kNumBsPerCoreTileFragement == 2, "Only for 16b gemm now.");

    // First convert 4b weight into fp16(weight + 16)
    weights2Half(weights, dest);

    ElementScale addon[kMmaIterationsB];
    if constexpr (kMmaIterationsB % 4 == 0) {
      const b64* scales_ptr = reinterpret_cast<const b64*>(scales.data());
      uint32_t* addon_ptr = reinterpret_cast<uint32_t*>(addon);
      if constexpr(kHasOffset){
        const uint32_t* p = reinterpret_cast<const uint32_t*>(offsets.data());
        CUTLASS_PRAGMA_UNROLL
        for (int n_idx = 0; n_idx < kMmaIterationsB; n_idx += 4){
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
          asm volatile(
            "{\n\t"
            "  .reg  .b32    rb0, rb1, rb2;\n"

            // offset from [d, c, b, a] --> [d, b, c, a]
            "  prmt.b32      rb2, %4, rb0, 0x3120;\n"

            // static_cast<cutlass::half_t>(-16 - offset)
            // input [d, b, c, a],
            "  shl.b32       rb0, rb2, 6;\n"     // rb0 = [x, b, x, a] << 6
            "  shr.u32       rb1, rb2, 2;\n"     // rb1 = [x, d, x, c] << 6
            "  lop3.b32      rb0, rb0, 0x03c003c0, 0xcc00cc00, 0xea;\n" // a & 0x03c0 | 0xcc00
            "  lop3.b32      rb1, rb1, 0x03c003c0, 0xcc00cc00, 0xea;\n"
            "  mul.rn.f16x2  %0, %2, rb0;\n"    // offset = scale * (-16 - offset)
            "  mul.rn.f16x2  %1, %3, rb1;\n"
            "}\n"
            : "=r"(addon_ptr[0]), "=r"(addon_ptr[1])
            : "r"(scales_ptr->pair.a), "r"(scales_ptr->pair.b),
              "r"(p[0]));
#else
          assert(0);
#endif
          scales_ptr++;
          p++;
          addon_ptr += 2;
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int n_idx = 0; n_idx < kMmaIterationsB; n_idx += 4){
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
          asm volatile(
            "{\n\t"
            "  .reg  .b32    rb0;\n"
            "  mov.u32       rb0, 0xce00ce00;\n"
            "  mul.rn.f16x2  %0, %2, rb0;\n"    // offset = scale * (-16 - 8)
            "  mul.rn.f16x2  %1, %3, rb0;\n"
            "}\n"
            : "=r"(addon_ptr[0]), "=r"(addon_ptr[1])
            : "r"(scales_ptr->pair.a), "r"(scales_ptr->pair.b));
#else
          assert(0);
#endif
          scales_ptr++;
          addon_ptr += 2;
        }
      }
    } else if constexpr (kMmaIterationsB % 2 == 0) {
      const uint32_t* scales_ptr = reinterpret_cast<const uint32_t*>(scales.data());
      uint32_t* addon_ptr = reinterpret_cast<uint32_t*>(addon);

      if constexpr (kHasOffset){
        // possible buffer over read 2 bytes here.
        const uint32_t* p = reinterpret_cast<const uint32_t*>(offsets.data());
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        asm volatile(
          "{\n\t"
          "  .reg  .b32    rb0, rb1, rb2;\n"

          // offset from [?, ?, b, a] --> [?, b, ?, a]
          "  prmt.b32      rb2, %2, rb0, 0x3120;\n"

          // static_cast<cutlass::half_t>(-16 - offset)
          // input [d, b, c, a],
          "  shl.b32       rb0, rb2, 6;\n"     // rb0 = [x, b, x, a] << 6
          "  lop3.b32      rb0, rb0, 0x03c003c0, 0xcc00cc00, 0xea;\n" // a & 0x03c0 | 0xcc00
          "  mul.rn.f16x2  %0, %1, rb0;\n"    // offset = scale * (-16 - offset)
          "}\n"
          : "=r"(addon_ptr[0])
          : "r"(scales_ptr[0])
            "r"(p[0]));
#else
        assert(0);
#endif
      } else {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
        asm volatile(
          "{\n\t"
          "  .reg  .b32    rb0;\n"
          "  mov.u32       rb0, 0xce00ce00;\n"
          "  mul.rn.f16x2  %0, %1, rb0;\n"    // offset = scale * (-16 - 8)
          "}\n"
          : "=r"(addon_ptr[0])
          : "r"(scales_ptr[0]));
#else
        assert(0);
#endif
      }
    } else {
      // kMmaIterationsB == 1
      if constexpr(kHasOffset){
        uint8_t zp = offsets[0];
        addon[0] = scales[0] * static_cast<ElementScale>(-16 - static_cast<int>(zp));
      } else {
        addon[0] = scales[0] * static_cast<ElementScale>(-16-8);
      }
    }

    int out_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int n_out = 0; n_out < kMmaIterationsB; n_out++){
      CUTLASS_PRAGMA_UNROLL
      for (int mma_tile_out_idx = 0; mma_tile_out_idx < kBTilesPerMma; mma_tile_out_idx++){
        dest[out_idx] = scales[n_out] * dest[out_idx] + addon[n_out];
        dest[out_idx + 1] = scales[n_out] * dest[out_idx + 1] + addon[n_out];
        out_idx += 2;
      }
    }
  }

  /// Advances the pointer
  CUTLASS_HOST_DEVICE
  QuantBMetaMmaTensorOpTileIterator &operator++() {
    // This is for operand B, so advance on the K dimension
    lane_position_ += make_Coord(MetaTile::TileShapeB::kRow, 0);
    return *this;
  }

  CUTLASS_DEVICE
  QuantBMetaMmaTensorOpTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int rows = tile_offset.row() * MetaTile::TileShapeB::kRow;
    int columns = tile_offset.column() * MetaTile::TileShapeB::kColumn;
    lane_position_ += TensorCoord(rows, columns);
    return *this;
  }

};


////////////////////////////////////////////////////////////////////////////////
}  // namespace warp
}  // namespace gemm
}  // namespace cutlass
