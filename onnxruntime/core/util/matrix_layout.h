/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    matrix_layout.h
 *
 * Abstract:
 *   Utils for simplifying positioning and striding in tensors. Inspired
 *   by CUTLASS, striving for 0 runtime cost while promote safety.
 *
 *   Only supports 2D tensors (matrix) for now.
 */

#pragma once

#include <cstdint>
#include "core/common/gsl.h"

// TODO!! Already have this in cuda, what about cpu code though?
#if defined(_MSC_VER)
#define ORT_FORCEINLINE __forceinline
#else
#define ORT_FORCEINLINE __attribute__((always_inline)) inline
#endif

namespace onnxruntime {

//
// Clang-format doesn't handle force inline decorator well, it insists on
// adding extra indentation to the next line, making it very confusing
// to read. So we turn it off for this file.
// clang-format off
//

/**
 * @brief A tuple of integers to represent tensor coordinates
 */
template <
    int Rank_,                     ///< Logical rank of coordinate
    typename Index_ = int,         ///< Index type used for each dimension
    typename LongIndex_ = int64_t  ///< Long index type used for linear offsets
    >
struct Position {
 public:
  /// Number of elements in Position
  static int const kRank = Rank_;

  /// Index type used to store elements
  using Index = Index_;

  /// Type used to represent linear offsets
  using LongIndex = LongIndex_;

 private:
  Index idx[kRank];

 public:
  ORT_FORCEINLINE explicit Position(Index value = Index(0)) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = value;
    }
  }

  /// Constructs from an array of integers
  ORT_FORCEINLINE
  Position(Index const (&_idx)[kRank]) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = _idx[i];
    }
  }

  template <int R, typename I, typename L>
  ORT_FORCEINLINE
  Position(Position<R, I, L> other) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = other[i];
    }
  }

  ORT_FORCEINLINE
  Position operator+(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] + b.idx[i];
    }
    return c;
  }

  ORT_FORCEINLINE
  Position operator-(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] - b.idx[i];
    }
    return c;
  }

  ORT_FORCEINLINE
  Position operator*(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] * b.idx[i];
    }
    return c;
  }

  ORT_FORCEINLINE
  Position operator/(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] / b.idx[i];
    }
    return c;
  }

  ORT_FORCEINLINE
  Position& operator+=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] += b.idx[i];
    }
    return *this;
  }

  ORT_FORCEINLINE
  Position& operator-=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] -= b.idx[i];
    }
    return *this;
  }

  ORT_FORCEINLINE
  Position& operator*=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] *= b.idx[i];
    }
    return *this;
  }

  ORT_FORCEINLINE
  Position& operator/=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] /= b.idx[i];
    }
    return *this;
  }

  ORT_FORCEINLINE Index& operator[](int dim) { return idx[dim]; }

  ORT_FORCEINLINE Index const& operator[](int dim) const { return idx[dim]; }

  ORT_FORCEINLINE bool operator==(Position const& b) const {
    bool equal = true;
    for (int i = 0; equal && i < kRank; ++i) {
      equal = (idx[i] == b.idx[i]);
    }
    return equal;
  }

  ORT_FORCEINLINE bool operator!=(Position const& b) const { return !(*this == b); }

  ORT_FORCEINLINE
  Position& clamp(Position const& max, Position const& min = Position()) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = std::max(std::min(idx[i], max.idx[i]), min.idx[i]);
    }
    return *this;
  }

  ORT_FORCEINLINE
  Index sum() const {
    Index sum_(idx[0]);
    for (int i = 1; i < kRank; ++i) {
      sum_ += idx[i];
    }
    return sum_;
  }

  ORT_FORCEINLINE
  LongIndex product() const {
    LongIndex product_(idx[0]);
    for (int i = 1; i < kRank; ++i) {
      product_ *= idx[i];
    }
    return product_;
  }
};

template <typename T, typename L = int64_t>
Position<2, T, L> make_Position(T _0, T _1) {
  T values[2] = {_0, _1};
  return Position<2, T, L>(values);
}

template <typename T, typename L = int64_t>
Position<3, T, L> make_Position(T _0, T _1, T _2) {
  T values[3] = {_0, _1, _2};
  return Position<2, T, L>(values);
}

/// Describes the size of a matrix tile
template <
    int Row_,    ///< rows of a matrix
    int Column_  ///< columns of a matrix
    >
struct MatrixShape {
  static int const kRow = Row_;              ///< rows of a matrix
  static int const kColumn = Column_;        ///< columns of a matrix
  static int const kCount = Row_ * Column_;  ///< total number of elements in a matrix

  ORT_FORCEINLINE static Position<2> toCoord() {
    return make_Position(kRow, kColumn);
  }
};

/**
 * @brief Defines a mapping from logical coordinate to linear memory
 * offsets in a row major layout matrix
 */
class RowMajorLayout {
 public:
  /// Index type used for coordinates
  using Index = int;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using MatCoord = Position<2, Index, LongIndex>;

 private:
  Index stride_;

 public:
  ORT_FORCEINLINE
  RowMajorLayout(Index ldm = 0) : stride_(ldm) {}

  ORT_FORCEINLINE static RowMajorLayout packed(MatCoord const& extent) {
    return RowMajorLayout(extent[1]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (row, column)
  ORT_FORCEINLINE
  LongIndex operator()(MatCoord const& coord) const {
    return LongIndex(coord[0]) * stride_ + coord[1];
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  ORT_FORCEINLINE
  MatCoord inverse(LongIndex offset) const {
    return make_Position(Index(offset / stride_), Index(offset % stride_));
  }

  ORT_FORCEINLINE
  Index stride() const {
    return stride_;
  }
};

class ColumnMajorLayout {
 public:
  /// Index type used for coordinates
  using Index = int;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using MatCoord = Position<2, Index, LongIndex>;

 private:
  Index stride_;

 public:
  ORT_FORCEINLINE
  ColumnMajorLayout(Index ldm = 0) : stride_(ldm) {}

  ORT_FORCEINLINE static ColumnMajorLayout packed(MatCoord const& extent) {
    return ColumnMajorLayout(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (row, column)
  ORT_FORCEINLINE
  LongIndex operator()(MatCoord const& coord) const {
    return LongIndex(coord[1]) * LongIndex(stride_) + coord[0];
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  ORT_FORCEINLINE
  MatCoord inverse(LongIndex offset) const {
    return make_Position(Index(offset % stride_), Index(offset / stride_));
  }

  ORT_FORCEINLINE
  Index stride() const {
    return stride_;
  }
};

/**
 * @brief A reference to a tensor, with a layout object to map logical
 * coordinates to linear offsets.
 */
template <
    /// Data type of element stored within tensor, must be numerical types
    typename Element_,
    /// Defines a mapping from logical coordinate to linear memory offsets
    typename Layout_,
    /// If true, extra bounds checking is performed on all accesses
    bool ExtraBoundsCheck_ = false>
class MatrixRef {
 public:
  /// Data type of individual access
  using Element = Element_;

  using Reference = Element&;

  /// Mapping function from logical coordinate to linear memory
  using Layout = Layout_;

  /// Index type
  using Index = typename Layout::Index;

  /// Long index used for pointer offsets
  using LongIndex = typename Layout::LongIndex;

  /// Coordinate in logical tensor space
  using MatCoord = typename Layout::MatCoord;

  /// MatrixRef to constant data
  using ConstMatrixRef = MatrixRef<
      typename std::remove_const<Element>::type const,
      Layout, ExtraBoundsCheck_>;

  /// MatrixRef to non-constant data
  using NonConstMatrixRef = MatrixRef<
      typename std::remove_const<Element>::type,
      Layout, ExtraBoundsCheck_>;

  static constexpr bool IsNonConstRef = std::is_same<NonConstMatrixRef, MatrixRef<Element_, Layout_>>::value;

 private:
  /// Pointer to data
  gsl::span<Element> data_;

  /// Shape of matrix
  MatCoord shape_;

  /// Layout object maps logical coordinates to linear offsets
  Layout layout_;

 public:
  ORT_FORCEINLINE
  MatrixRef() : data_() {}

  ORT_FORCEINLINE
  MatrixRef(
      gsl::span<Element> const& data,  ///< pointer to start of tensor
      MatCoord const& shape            ///< shape of tensor
      ) : data_(data), shape_(shape), layout_(Layout::packed(shape)) {
    Expects(data_.size() >= size_t(shape_.product()));
  }

  ORT_FORCEINLINE
  MatrixRef(
      Element* ptr,          ///< pointer to start of tensor
      LongIndex size,        ///< size of tensor in elements
      MatCoord const& shape  ///< shape of tensor
      ) : data_(ptr, size), shape_(shape), layout_(Layout::packed(shape)) {
    Expects(data_.size() >= shape_.product());
  }

  /// Converting constructor from MatrixRef to non-constant data.
  template <typename _Magic = int>
  ORT_FORCEINLINE
  MatrixRef(
      NonConstMatrixRef const& ref,  ///< MatrixRef to non-const data
      /// SFINAE trick to avoid creating a copy-constructor when Element_ is already non-const
      _Magic magic = (typename std::enable_if<!IsNonConstRef, _Magic>::type)0
      ) : data_(ref.data()), shape_(ref.shape()), layout_(Layout::packed(ref.shape())) {}

  ORT_FORCEINLINE
  ConstMatrixRef const_ref() const {
    return ConstMatrixRef(data_, shape_);
  }

  ORT_FORCEINLINE
  NonConstMatrixRef non_const_ref() {
    return NonConstMatrixRef(
        const_cast<typename std::remove_const<Element>::type*>(data_.data()),
        data_.size(), shape_);
  }

  /// Returns true if the MatrixRef is non-null
  ORT_FORCEINLINE
  bool good() const { return !data_.empty(); }

  ORT_FORCEINLINE
  gsl::span<Element> const& data() const { return data_; }

  ORT_FORCEINLINE
  MatCoord const& shape() const { return shape_; }

  ORT_FORCEINLINE
  Layout& layout() { return layout_; }

  ORT_FORCEINLINE
  Layout layout() const { return layout_; }

  ORT_FORCEINLINE
  Index stride() const { return layout_.stride(); }

  ORT_FORCEINLINE
  Index& stride() { return layout_.stride(); }

  /// Computes the offset of an index from the origin of the tensor
  ORT_FORCEINLINE
  LongIndex offset(MatCoord const& coord) const {
    if constexpr (ExtraBoundsCheck_) {
      Expects(coord[0] >= 0 && coord[0] < shape_[0]);
      Expects(coord[1] >= 0 && coord[1] < shape_[1]);
    }
    return layout_(coord);
  }

  /// Returns a reference to the element at a given Coord
  ORT_FORCEINLINE
  Reference at(MatCoord const& coord) const {
    return data_[offset(coord)];
  }

  ORT_FORCEINLINE
  Reference at(int row, int col) const {
    return data_[offset(make_Position(row, col))];
  }

  /// Returns a reference to the element at a given Coord
  ORT_FORCEINLINE
  Reference operator[](MatCoord const& coord) const {
    return data_[offset(coord)];
  }
};

/// Constructs a MatrixRef, deducing types from arguments.
template <
    typename Element,
    typename Layout = RowMajorLayout,
    bool ExtraBoundsCheck = false>
ORT_FORCEINLINE
MatrixRef<Element, Layout, ExtraBoundsCheck>
make_MatrixRef(
    Element* ptr,
    int64_t size,
    typename Layout::MatCoord const& shape) {
  return MatrixRef<Element, Layout, ExtraBoundsCheck>(ptr, size, shape);
}

template <
    typename Element,
    typename Layout = RowMajorLayout,
    bool ExtraBoundsCheck = false>
ORT_FORCEINLINE
MatrixRef<Element, Layout, ExtraBoundsCheck>
make_MatrixRef(
    const gsl::span<Element>& span,
    typename Layout::MatCoord const& shape) {
  return MatrixRef<Element, Layout, ExtraBoundsCheck>(span, shape);
}

// clang-format off

}  // namespace onnxruntime
