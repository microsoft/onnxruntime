/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * @file optional_predicated_tile_access_iter.h
 * @brief Templates for loading and storing optional tiles of matrix data.
 *   This iterator is just a wrapper of PredicatedTileAccessIterator, with
 *   the option to turn it off at compile time and minimize its runtime
 *   footprint. Also, it utilize the higher numbered threads in the
 *   threadblock when  the iterator can not utilize all the threads.
 */

#pragma once

#include <variant>

#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {


////////////////////////////////////////////////////////////////////////////////

/// Optional 2-D matrix data loader, when element is std::monostate, the
/// iterator becomes no-op with minimal runtime footprint. Also, it utilize the
/// higher numbered threads in the threadblock when the iterator can not utilize
/// all the threads.
///
template <
    /// Tile shape of the iterator
    typename Shape_,
    /// Element data type of the iterator, no-op when it is std::monostate
    typename Element_,
    /// Layout of the source matrix
    typename Layout_,
    int AdvanceRank_,
    typename ThreadMap_,
    typename AccessType_,
    /// Number of threads in the threadblock, when provided, the iterator
    /// will utilize the higher numbered threads
    int kThreadBlockSize_ = -1>
class OptionalPredicatedTileAccessIterator{
 public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  static constexpr int kAdvanceRank = AdvanceRank_;
  static constexpr int kThreadblockSize = kThreadBlockSize_;

  static_assert(!std::is_same<Element, std::monostate>::value,
      "Disabled Iterator failed to match the specialized version below.");
  static_assert(kThreadblockSize == -1 || kThreadblockSize >= ThreadMap::kThreads,
      "kThreadblockSize must be no smaller than ThreadMap::kThreads");

  using Base = PredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank, ThreadMap, AccessType>;

  using LongIndex = typename Base::LongIndex;
  using Mask = typename Base::Mask;
  using TensorCoord = typename Base::TensorCoord;
  using TensorRef = typename Base::TensorRef;
  using Params = typename Base::Params;
  using Pointer = typename Base::Pointer;

  static constexpr int kAccessesPerVector = Base::kAccessesPerVector;

  CUTLASS_HOST_DEVICE
  static int flip_thread_id(int thread_id){
    if constexpr (kThreadblockSize > 0) {
      return kThreadblockSize - 1 - thread_id;
    }
    return thread_id;
  }

 public:
   Base base_;

  /// Default constructor
  OptionalPredicatedTileAccessIterator(): base_() {};

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  OptionalPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : base_(params, pointer, extent, flip_thread_id(thread_id), threadblock_offset) {}

  /// Construct a PredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  OptionalPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id)
      : OptionalPredicatedTileAccessIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) {}

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

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_DEVICE
  void add_tile_offset(
      TensorCoord const &tile_offset) {
    base_.add_tile_offset(tile_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return base_.get();
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  OptionalPredicatedTileAccessIterator &operator++() {
    ++base_;
    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  OptionalPredicatedTileAccessIterator operator++(int) {
    OptionalPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    base_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    base_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    base_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    base_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return base_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for the disabled version
/// Reduce runtime overhead
///
template <
    /// Tile shape of the iterator
    typename Shape_,
    typename Layout_,
    int AdvanceRank_,
    typename ThreadMap_,
    typename AccessType_,
    int kThreadBlockSize_>
class OptionalPredicatedTileAccessIterator<Shape_, std::monostate, Layout_, AdvanceRank_, ThreadMap_, AccessType_, kThreadBlockSize_>{
 public:

  using Shape = Shape_;
  using Element = std::monostate;
  using Layout = Layout_;
  static int const kAdvanceRank = AdvanceRank_;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  static constexpr int kThreadblockSize = kThreadBlockSize_;

  using Base = PredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank, ThreadMap, AccessType>;

  using LongIndex = typename Base::LongIndex;
  using Mask = typename Base::Mask;
  using TensorCoord = typename Base::TensorCoord;
  using TensorRef = typename Base::TensorRef;
  using Params = typename Base::Params;
  using Pointer = typename Base::Pointer;

  static constexpr int kAccessesPerVector = Base::kAccessesPerVector;

 public:
  std::monostate base_;

  /// Default constructor
  OptionalPredicatedTileAccessIterator(): base_() {};

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  OptionalPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : base_() {}

  /// Construct a PredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  OptionalPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id)
      : base_() {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {}

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_DEVICE
  void add_tile_offset(
      TensorCoord const &tile_offset) {}

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return nullptr;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  OptionalPredicatedTileAccessIterator &operator++() {
    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  OptionalPredicatedTileAccessIterator operator++(int) {
    return *this;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {}

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {}

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {}

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {}

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() const { return false; }
};

////////////////////////////////////////////////////////////////////////////////
}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass
