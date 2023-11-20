/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    matrix_layout.h

Abstract:

    Utils for simplifying positioning and striding in tensors. Inspired
    by CUTLASS, striving for 0 runtime cost while promote safety.

    Only supports 2D tensors (matrix) for now.

--*/

#include <cstdint>
#include "core/common/gsl.h"

// TODO!! Already have this in cuda, what about cpu code though?
#if defined(_MSC_VER)
#define __forceinline__ __forceinline
#else
#define __forceinline__ __attribute__((always_inline)) inline
#endif

namespace onnxruntime {

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
  __forceinline__ explicit Position(Index value = Index(0)) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = value;
    }
  }

  /// Constructs from an array of integers
  __forceinline__
  Position(Index const (&_idx)[kRank]) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = _idx[i];
    }
  }

  template <int R, typename I, typename L>
  __forceinline__
  Position(Position<R, I, L> other) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = other[i];
    }
  }

  __forceinline__
      Position
      operator+(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] + b.idx[i];
    }
    return c;
  }

  __forceinline__
      Position
      operator-(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] - b.idx[i];
    }
    return c;
  }

  __forceinline__
      Position
      operator*(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] * b.idx[i];
    }
    return c;
  }

  __forceinline__
      Position
      operator/(Position const& b) const {
    Position c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] / b.idx[i];
    }
    return c;
  }

  __forceinline__
      Position&
      operator+=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] += b.idx[i];
    }
    return *this;
  }

  __forceinline__
      Position&
      operator-=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] -= b.idx[i];
    }
    return *this;
  }

  __forceinline__
      Position&
      operator*=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] *= b.idx[i];
    }
    return *this;
  }

  __forceinline__
      Position&
      operator/=(Position const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] /= b.idx[i];
    }
    return *this;
  }

  __forceinline__ Index& operator[](int dim) { return idx[dim]; }

  __forceinline__ Index const& operator[](int dim) const { return idx[dim]; }

  __forceinline__ bool operator==(Position const& b) const {
    bool equal = true;
    for (int i = 0; equal && i < kRank; ++i) {
      equal = (idx[i] == b.idx[i]);
    }
    return equal;
  }

  __forceinline__ bool operator!=(Position const& b) const { return !(*this == b); }

  __forceinline__
  Position& clamp(Position const& max, Position const& min = Position()) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = __NV_STD_MAX(__NV_STD_MIN(idx[i], max.idx[i]), min.idx[i]);
    }
    return *this;
  }

  __forceinline__
  Index sum() const {
    Index sum_(idx[0]);
    for (int i = 1; i < kRank; ++i) {
      sum_ += idx[i];
    }
    return sum_;
  }

  __forceinline__
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

  __forceinline__ static Position<2> toCoord() {
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
  __forceinline__
  RowMajorLayout(Index ldm = 0) : stride_(ldm) {}

  __forceinline__ static RowMajorLayout packed(MatCoord const& extent) {
    return RowMajorLayout(extent[1]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (row, column)
  __forceinline__
  LongIndex operator()(MatCoord const& coord) const {
    return LongIndex(coord[0]) * stride_ + coord[1];
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  __forceinline__
  MatCoord inverse(LongIndex offset) const {
    return make_Position(Index(offset / stride_), Index(offset % stride_));
  }

  __forceinline__
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
  __forceinline__
  ColumnMajorLayout(Index ldm = 0) : stride_(ldm) {}

  __forceinline__ static ColumnMajorLayout packed(MatCoord const& extent) {
    return ColumnMajorLayout(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (row, column)
  __forceinline__
  LongIndex operator()(MatCoord const& coord) const {
    return LongIndex(coord[1]) * LongIndex(stride_) + coord[0];
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  __forceinline__
  MatCoord inverse(LongIndex offset) const {
    return make_Position(Index(offset % stride_), Index(offset / stride_));
  }

  __forceinline__
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

 private:
  /// Pointer to data
  gsl::span<Element> data_;

  /// Shape of matrix
  MatCoord shape_;

  /// Layout object maps logical coordinates to linear offsets
  Layout layout_;

 public:
  __forceinline__
  MatrixRef() : data_() {}

  __forceinline__
  MatrixRef(
      gsl::span<Element> const& data,  ///< pointer to start of tensor
      MatCoord const& shape            ///< shape of tensor
      ) : data_(data), shape_(shape), layout_(Layout::packed(shape)) {
    Expects(data_.size() >= size_t(shape_.product()));
  }

  __forceinline__
  MatrixRef(
      Element* ptr,          ///< pointer to start of tensor
      LongIndex size,        ///< size of tensor in elements
      MatCoord const& shape  ///< shape of tensor
      ) : data_(ptr, size), shape_(shape), layout_(Layout::packed(shape)) {
    Expects(data_.size() >= shape_.product());
  }

  /// Converting constructor from MatrixRef to non-constant data.
  template <typename _Magic = int>
  __forceinline__
  MatrixRef(
      NonConstMatrixRef const& ref,  ///< MatrixRef to non-const data
      /// SFINAE trick to avoid creating a copy-constructor when Element_ is already non-const
      _Magic magic =
          (typename std::enable_if<!std::is_same<NonConstMatrixRef, MatrixRef<Element_, Layout_>>::value, _Magic>::type)0) : data_(ref.data()), shape_(ref.shape()), layout_(Layout::packed(ref.shape())) {}

  __forceinline__
  ConstMatrixRef const_ref() const {
    return ConstMatrixRef(data_, shape_);
  }

  __forceinline__
  NonConstMatrixRef non_const_ref() const {
    return NonConstMatrixRef(
        const_cast<typename std::remove_const<Element>::type*>(data_.data()),
        data_.size(), shape_);
  }

  /// Returns true if the MatrixRef is non-null
  __forceinline__ bool good() const {
    return !data_.empty();
  }

  __forceinline__
  gsl::span<Element> const& data() const { return data_; }

  __forceinline__
  MatCoord const& shape() const { return shape_; }

  __forceinline__
  Layout& layout() { return layout_; }

  __forceinline__
  Layout layout() const { return layout_; }

  __forceinline__
  Index stride() const { return layout_.stride(); }

  __forceinline__
  Index& stride() { return layout_.stride(); }

  /// Computes the offset of an index from the origin of the tensor
  __forceinline__
  LongIndex offset(MatCoord const& coord) const {
    if constexpr (ExtraBoundsCheck_) {
      Expects(coord[0] >= 0 && coord[0] < shape_[0]);
      Expects(coord[1] >= 0 && coord[1] < shape_[1]);
    }
    return layout_(coord);
  }

  /// Returns a reference to the element at a given Coord
  __forceinline__
  Reference at(MatCoord const& coord) const {
    return data_[offset(coord)];
  }

  __forceinline__
  Reference at(int row, int col) const {
    return data_[offset(make_Position(row, col))];
  }

  /// Returns a reference to the element at a given Coord
  __forceinline__
  Reference operator[](MatCoord const& coord) const {
    return data_[offset(coord)];
  }
};

/// Constructs a MatrixRef, deducing types from arguments.
template <
    typename Element,
    typename Layout = RowMajorLayout,
    bool ExtraBoundsCheck = false>
__forceinline__
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
__forceinline__
MatrixRef<Element, Layout, ExtraBoundsCheck>
make_MatrixRef(
    const gsl::span<Element>& span,
    typename Layout::MatCoord const& shape) {
  return MatrixRef<Element, Layout, ExtraBoundsCheck>(span, shape);
}

// Converting cutlass tensor to MatrixRef
//
//
// template <
//   typename Element,
//   typename LayoutCutlass,
//   typename Layout = std::conditional_t<std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value, ColumnMajorLayout, RowMajorLayout>
//   >
// __forceinline__
// MatrixRef<Element, Layout> make_MatrixRef(cutlass::HostTensor<Element, LayoutCutlass> const& tensor) {
//   static_assert(std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value
//                 || std::is_same<LayoutCutlass, cutlass::layout::RowMajor>::value);
//   auto shape = make_Position(tensor.extent().row(), tensor.extent().column());
//   auto* ptr = const_cast<typename std::remove_const<Element>::type *>(tensor.host_data());
//   return MatrixRef<Element, Layout>(ptr, tensor.capacity(), shape);
// }

// template <
//   typename Element,
//   typename LayoutCutlass,
//   typename Layout = std::conditional_t<std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value, ColumnMajorLayout, RowMajorLayout>
//   >
// __forceinline__
// MatrixRef<Element const, Layout> make_ConstMatrixRef(cutlass::HostTensor<Element, LayoutCutlass> const& tensor) {
//   static_assert(std::is_same<LayoutCutlass, cutlass::layout::ColumnMajor>::value
//                 || std::is_same<LayoutCutlass, cutlass::layout::RowMajor>::value);
//   auto shape = make_Position(tensor.extent().row(), tensor.extent().column());
//   return MatrixRef<Element const, Layout>(tensor.host_data(), tensor.capacity(), shape);
// }

}  // namespace onnxruntime
