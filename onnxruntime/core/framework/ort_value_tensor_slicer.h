// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iterator>
#include <limits>
#include <type_traits>

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#endif

namespace onnxruntime {

/**
Class to provide a slicing service over a Tensor stored within an OrtValue with shape
{batch size, sequence length, <input shape>}. Access to the slices is via an iterator interface.

For each iteration an OrtValue will be returned containing a sub-Tensor of the original Tensor.
The sub-Tensor applies the relevant offset to the data address from the original Tensor in order
to avoid any memory allocations/copies for the tensor data.
*/
template <typename T>
class OrtValueTensorSlicer {
 public:
  /**
  Create a new instance to slice the Tensor contained in an MLValue
  into sub-Tensors contained within new OrtValue instances that are accessed via the Iterator.
  T must be 'OrtValue' or 'const OrtValue'
    @param slice_dimension Dimension to slice on.
    @param dim0_offset Offset to start at. Only meaningful if slice_dimension != 0.
           e.g. if input is [batch, seq_len, data] and you want to slice the seq_len dimension, you need to
                create an Iterator instance for each batch item, incrementing dim0_offset for each one.
  */
  static OrtValueTensorSlicer Create(T& ort_value, int64_t slice_dimension = 0, int64_t dim0_offset = 0);

  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    enum class Direction { kForward,
                           kReverse };

    explicit Iterator(T& ort_value, size_t slice_dimension, size_t dim0_offset, int64_t position,
                      Direction direction = Direction::kForward);

    bool operator==(const Iterator& other) const noexcept {
      return ort_value_ == other.ort_value_ && position_ == other.position_;
    }

    bool operator!=(const Iterator& other) const noexcept { return !(*this == other); }

    Iterator& operator++() {
      position_ += increment_by_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp{*this};
      ++(*this);

      return tmp;
    }

    Iterator& operator+=(difference_type n) {
      position_ += increment_by_ * n;
      return *this;
    }

    // const accessor is always enabled
    const_reference operator*() const {
      ORT_ENFORCE(position_ >= 0 && position_ < sequence_length_);
      if (position_ != position_materialized_) {
        MaterializeMLValue();
      }

      return current_;
    }

    // non-const is only enabled if T is not const (i.e. is 'OrtValue' not 'const OrtValue')
    typename std::enable_if<!std::is_const<reference>::value, reference>::type operator*() {
      ORT_ENFORCE(position_ >= 0 && position_ < sequence_length_);
      if (position_ != position_materialized_) {
        MaterializeMLValue();
      }

      return current_;
    }

    virtual ~Iterator() = default;

   private:
    // virtual so scenarios where the void* in OrtValue::data_ isn't just a raw pointer to data (e.g. it's a handle)
    // can implement the correct handling
    virtual void MaterializeMLValue() const;

    T* ort_value_;
    int64_t position_;

    // 1 for forward, -1 for reverse
    // Alternatively we could apply a std::reverse_iterator adapter to Iterator, however the primary use case
    // for this class involves passing a mix of forward/reverse iterator instances in a single collection so
    // we need to handle the direction internally so only one type is involved in that collection.
    const int64_t increment_by_;

    const void* tensor_data_raw_;
    MLDataType tensor_data_type_;
    const OrtMemoryInfo* tensor_location_;

    int64_t sequence_length_;
    TensorShape per_iteration_shape_;
    size_t per_iteration_offset_;

    mutable int64_t position_materialized_;  // position_ when current_ was created
    mutable OrtValue current_;
  };

  Iterator begin() const noexcept { return Iterator(*ort_value_, static_cast<size_t>(slice_dimension_), static_cast<size_t>(dim0_offset_), 0); }
  Iterator end() const noexcept {
    return Iterator(*ort_value_, static_cast<size_t>(slice_dimension_), static_cast<size_t>(dim0_offset_), std::numeric_limits<int64_t>::max());
  }

  Iterator rbegin() const noexcept {
    return Iterator(*ort_value_, static_cast<size_t>(slice_dimension_), static_cast<size_t>(dim0_offset_), std::numeric_limits<int64_t>::max(),
                    Iterator::Direction::kReverse);
  }

  Iterator rend() const noexcept {
    return Iterator(*ort_value_, static_cast<size_t>(slice_dimension_), static_cast<size_t>(dim0_offset_), -1, Iterator::Direction::kReverse);
  }

 private:
  OrtValueTensorSlicer(T& ort_value, int64_t slice_dimension, int64_t dim0_offset) noexcept
      : ort_value_(&ort_value), slice_dimension_(slice_dimension), dim0_offset_(dim0_offset) {}

  T* ort_value_;
  int64_t slice_dimension_;
  int64_t dim0_offset_;
};

}  // namespace onnxruntime
