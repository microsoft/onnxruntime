/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * @file optional_regular_tile_access_iter.h
 * @brief Templates implementing the address computation of storing of tiles
 *   from pitch-linear rank=2 tensors.
 *
 *   This iterator is just a wrapper of RegularTileAccessIterator, with the
 *   option to turn it off at compile time and minimize its runtime footprint.
 *   Also, it utilize the higher numbered threads in the threadblock when the
 *   iterator can not utilize all the threads.
 *
 *   Must be used in conjunction with OptionalPredicatedTileAccessIterator,
 *   with the same template parameters.
 */

#pragma once

#include <variant>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Optional 2-D tile iterator, when element is std::monostate, the iterator
/// becomes no-op with minimal runtime footprint. Also, it utilize the higher
/// numbered threads in the threadblock when the iterator can not utilize all
/// the threads.
///
template <
    /// Tile shape of the iterator
    typename Shape_,
    typename Element_,
    typename Layout_,
    int AdvanceRank,
    typename ThreadMap_,
    /// Number of threads in the threadblock, when not -1, the iterator
    /// will utilize the higher numbered threads
    int ThreadblockSize_ = -1,
    int Alignment =
        sizeof_bits<Element_>::value * ThreadMap_::kElementsPerAccess / 8>
class OptionalRegularTileAccessIterator{
 public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  using ThreadMap = ThreadMap_;
  static constexpr int kAlignment = Alignment;
  static constexpr int kThreadblockSize = ThreadblockSize_;

  static_assert(!std::is_same<Element, std::monostate>::value,
      "Disabled Iterator failed to match the specialized template");
  static_assert(kThreadblockSize == -1 || kThreadblockSize >= ThreadMap::kThreads,
      "kThreadblockSize must be no smaller than ThreadMap::kThreads");

  using Base = RegularTileAccessIterator<Shape, Element, Layout, AdvanceRank, ThreadMap, Alignment>;

  using LongIndex = typename Base::LongIndex;
  using TensorRef = typename Base::TensorRef;
  using TensorCoord = typename Base::TensorCoord;
  using AccessType = typename Base::AccessType;

  CUTLASS_HOST_DEVICE
  static int flip_thread_id(int thread_id){
    if constexpr (kThreadblockSize > 0) {
      return kThreadblockSize - 1 - thread_id;
    }
    return thread_id;
  }

 private:

  Base base_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  OptionalRegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : base_(ref, flip_thread_id(thread_id)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    base_.set_iteration_index(index);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    base_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() const {
    return base_.get();
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  OptionalRegularTileAccessIterator &operator++() {
    ++base_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  OptionalRegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset in the unit of tile.
  /// In GEMM/Conv implementation, this is used to move in the k dimension in the shared memory.
  /// Below layouts are the shared memory layouts.  Current SM50 SIMT kernels only use col major A and row major B.
  ///   For row major A operand, k dimension is contiguous dimension;
  ///   For col major A operand, k dimension is strided dimension;
  ///   For row major B operand, k dimension is strided dimension;
  ///   For col major B operand, k dimension is contiguous dimension.
  /// Below two classes map col/row major to the pitch linear coordinates used
  /// in this base class.
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    base_.add_tile_offset(coord);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization when Element is std::monostate, the iterator becomes no-op
///
template <
    typename Shape_,
    typename Layout_,
    int AdvanceRank,
    typename ThreadMap_,
    int ThreadblockSize_,
    int Alignment>
class OptionalRegularTileAccessIterator<Shape_, std::monostate, Layout_,
    AdvanceRank, ThreadMap_, ThreadblockSize_, Alignment>{
 public:

  using Shape = Shape_;
  using Element = std::monostate;
  using Layout = Layout_;
  using ThreadMap = ThreadMap_;
  static constexpr int kAlignment = Alignment;
  static constexpr int kThreadblockSize = ThreadblockSize_;

  using Base = RegularTileAccessIterator<Shape, Element, Layout, AdvanceRank, ThreadMap, Alignment>;

  using LongIndex = typename Base::LongIndex;
  using TensorRef = typename Base::TensorRef;
  using TensorCoord = typename Base::TensorCoord;
  using AccessType = typename Base::AccessType;

 private:

  std::monostate base_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  OptionalRegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : base_() {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {}

  /// Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() const {
    return nullptr;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  OptionalRegularTileAccessIterator &operator++() {
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  OptionalRegularTileAccessIterator operator++(int) {
    return *this;
  }

  /// Adds a tile offset in the unit of tile.
  /// In GEMM/Conv implementation, this is used to move in the k dimension in the shared memory.
  /// Below layouts are the shared memory layouts.  Current SM50 SIMT kernels only use col major A and row major B.
  ///   For row major A operand, k dimension is contiguous dimension;
  ///   For col major A operand, k dimension is strided dimension;
  ///   For row major B operand, k dimension is strided dimension;
  ///   For col major B operand, k dimension is contiguous dimension.
  /// Below two classes map col/row major to the pitch linear coordinates used
  /// in this base class.
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {}
};

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass
