// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/gsl.h"
#include "core/common/narrow.h"

#ifndef SHARED_PROVIDER
#include "core/framework/utils.h"
#endif
#include "core/common/safeint.h"
namespace onnxruntime {

struct TensorPitches : TensorShapeVector {
  TensorPitches(const Tensor& tensor, size_t rank = 0) : TensorPitches(tensor.Shape(), rank) {}
  TensorPitches(const TensorShape& shape, size_t rank = 0) : TensorPitches(shape.GetDims(), rank) {}
  TensorPitches(const TensorShapeVector& dims, size_t rank = 0)
      : TensorShapeVector(std::max(rank, dims.size()), 0) {
    Calculate(gsl::span<int64_t>(data(), size()), gsl::make_span(dims));
  }
  TensorPitches(const gsl::span<const int64_t>& dims, size_t rank = 0)
      : TensorShapeVector(std::max(rank, dims.size()), 0) {
    Calculate(gsl::span<int64_t>(data(), size()), dims);
  }

  static bool Calculate(const gsl::span<int64_t>& p, const gsl::span<const int64_t>& dims) {
    // The pitches is the size of the next inner axis. Aka the amount to move by one of the next inner axis.
    // For a tensor with shape(2,3,4,5) the values would be: (3*4*5, 4*5, 5, 1)
    // Note that the outermost '2' is never used, as you never need to move by the entire size of the outermost axis

    auto tensor_rank = dims.size();
    auto pitch_rank = p.size();
    auto padded_rank = pitch_rank - tensor_rank;
    if (static_cast<ptrdiff_t>(padded_rank) < 0)
      return false;

    // Guard against Scalars
    if (pitch_rank == 0) {
      return true;
    }

    *(p.rbegin()) = 1;  // The innermost axis is 1 (single values)
    if (tensor_rank > 1) {
      for (size_t i = tensor_rank - 1; i-- > 0;) {
        p.operator[](i + padded_rank) = p.operator[](i + 1 + padded_rank) * dims[i + 1];
      }
    }

    if (padded_rank >= 1) {
      for (size_t i = 0; i < padded_rank; ++i) {
        if (i == 0 && tensor_rank > 0)  // For scalar tensor, the values in the pitches are all 1.
          p.operator[](padded_rank - 1) = p.operator[](padded_rank) * dims[0];
        else
          p.operator[](padded_rank - 1 - i) = p.operator[](padded_rank - 1);
      }
    }
    return true;
  }
};

// This class is to iterate through the axes of an arbitrarily shaped tensor
// For example, a tensor with shape (2,3,4) will be iterated in this order:
// (0,0,x) (0,1,x) (0,2,x) (1,0,x) (1,1,x) (1,2,x)
// Note: The innermost axis is not iterated over since it's always special cased
struct TensorAxisCounters {
  TensorAxisCounters(const Tensor& tensor) : tensor_(tensor) {
    indices_.resize(tensor_.Shape().NumDimensions() - 1, 0);
    axis_ = indices_.size();

    // If a tensor has a shape, but one of the axes is 0 in size, there are no elements, so nothing to iterate
    if (tensor_.Shape().Size() == 0)
      running_ = false;
  }

  // Returns true if there was a carry to the next axis
  bool Increment() {
    if (axis_-- == 0) {
      running_ = false;
      return false;
    }

    if (++indices_[axis_] != tensor_.Shape()[axis_]) {
      axis_ = indices_.size();
      return false;
    }

    indices_[axis_] = 0;  // Reset the counter for this axis
    return true;          // There was a carry
  }

  size_t Axis() const { return axis_; }
  operator bool() const { return running_; }

 private:
  const Tensor& tensor_;
  bool running_{true};
  size_t axis_;
  TensorShapeVector indices_;  // There is no index for innermost axis since it's a special case
};

struct ExtentAxisCounters {
  ExtentAxisCounters(gsl::span<const int64_t> extents) : extents_(extents) {
    indices_.resize(extents_.size() - 1, 0);
    axis_ = indices_.size();

    // If a tensor has a shape, but one of the axes is 0 in size, there are no elements, so nothing to iterate
    if (std::find(extents.begin(), extents.end(), 0) != extents.end())
      running_ = false;
  }

  // Returns true if there was a carry to the next axis
  bool Increment() {
    if (axis_-- == 0) {
      running_ = false;
      return false;
    }

    if (++indices_[axis_] != extents_[axis_]) {
      axis_ = indices_.size();
      return false;
    }

    indices_[axis_] = 0;  // Reset the counter for this axis
    return true;          // There was a carry
  }

  size_t Axis() const { return axis_; }
  operator bool() const { return running_; }

 private:
  bool running_{true};
  size_t axis_;
  TensorShapeVector indices_;         // There is no index for innermost axis since it's a special case
  gsl::span<const int64_t> extents_;  // The extents of each axis
};

// A std::vector that holds the number of entries to skip to go to the next axis start given an extent
// and optionally steps along each axis:
// This is used by the SliceIterator to iterate over a slice of a tensor
struct SliceSkips : TensorShapeVector {
  SliceSkips(const TensorShape& input_shape, gsl::span<const int64_t> extents, gsl::span<const int64_t> steps)
      : TensorShapeVector(input_shape.NumDimensions(), 0) {
    auto dims = input_shape.GetDims();
    ORT_ENFORCE(dims.size() == extents.size() &&
                dims.size() >= steps.size());

    ptrdiff_t inner_most_dim = dims.size() - 1;
    // assume step == 1 if not present
    ptrdiff_t steps_size = steps.size();
    int64_t steps_i = 1;
    if (inner_most_dim >= 0 && inner_most_dim < steps_size)
      steps_i = steps[inner_most_dim];

    SafeInt<ptrdiff_t> pitch = 1;
    for (size_t i = size(); i-- > 0;) {
      auto prevPitch = pitch;
      pitch *= static_cast<ptrdiff_t>(dims[i]);

      // assume step == 1 if not present
      int64_t steps_i_minus_1 = 1;
      if (i > 0 && static_cast<ptrdiff_t>(i) - 1 < steps_size)
        steps_i_minus_1 = steps[i - 1];

      // first "revert" back to the old starting position (term with -ve sign)
      // and then "step" over the pitch accordingly (term with +ve sign)
      operator[](i) = steps_i_minus_1 * pitch - steps_i * extents[i] * prevPitch;

      steps_i = steps_i_minus_1;
    }
  }
};

// This provides easy sequential iteration over a subset of a tensor given a span of starts, extents & optionally steps
// The base class is type agnostic to minimize binary size. The derived class provides any type specific logic.
struct SliceIteratorBase {
 private:
  enum class byte : unsigned char {};

  // Initialize initial skip and inner_extent.
  void Init(gsl::span<const int64_t> dims, gsl::span<const int64_t> starts, gsl::span<const int64_t> steps) {
    const auto dims_size = dims.size();
    ORT_ENFORCE(dims_size == starts.size() &&
                dims_size == extents_.size() &&
                dims_size >= steps.size());

    SafeInt<size_t> pitch = 1;
    // Initial skip, so that input_ points to the first element to copy
    for (size_t i = dims_size; i-- > 0;) {
      input_ += pitch * starts[i] * element_size_;
      pitch *= static_cast<size_t>(dims[i]);
    }

    inner_extent_ = static_cast<size_t>(extents_[dims_size - 1]);
    const auto steps_size = steps.size();
    // It could be -1
    inner_step_ = static_cast<ptrdiff_t>(dims_size == steps_size
                                             ? steps[steps_size - 1]
                                             : 1);

    // This block serves to maximize the amount of data that is copied
    // in one shot when trailing axis are not being sliced and, therefore,
    // more data can be copied in one shot. The inner dimension with step = 1
    // can always copy with more than one element unless its extent is 1.
    // However, when more than one trailing dimensions have steps one and copied
    // in its entirety, we can copy bigger blocks at the same time and start skipping
    // them as the skips are expected to be zero then for them.
    SafeInt<int64_t> max_copyable_elements = (inner_step_ == 1) ? inner_extent_ : 1;
    last_batching_axis_ = dims_size - 1;

    // Check if inner dimension is copied as a block in its entirety
    if (dims_size > 1 && inner_step_ == 1 && inner_extent_ == narrow<size_t>(dims[dims_size - 1])) {
      for (size_t dim = dims_size - 2;; dim--) {
        if (dim < steps_size && steps[dim] != 1) {
          break;
        }
        max_copyable_elements *= extents_[dim];
        last_batching_axis_ = dim;
        if (extents_[dim] != dims[dim]) {
          break;
        }
        if (dim == 0) {
          break;
        }
      }
    }
    max_copying_elements_block_ = max_copyable_elements;
  }

 protected:
  SliceIteratorBase(const Tensor& tensor, gsl::span<const int64_t> starts,
                    gsl::span<const int64_t> extents, gsl::span<const int64_t> steps)
      : is_string_tensor_(tensor.IsDataTypeString()),
        input_{reinterpret_cast<const byte*>(tensor.DataRaw())},
        element_size_(tensor.DataType()->Size()),
        extents_(extents),
        skips_(tensor.Shape(), extents, steps),
        indices_(extents.size(), 0) {
    auto dims = tensor.Shape().GetDims();
    Init(dims, starts, steps);
  }

  // This construct takes a explicit tensor_shape which might be different from the shape defined in input tensor.
  // The explicit tensor_shape usually has inner most axis flattened. For example, given shape[1,4,4,2], if last axis
  // does not have padding or slice, then it will be flattened as [1,4,8] for better performance (One inner most copy instead of 4).
  // Also supports arbitrary positive and negative stepping along individual axes
  SliceIteratorBase(const Tensor& tensor, const TensorShape& tensor_shape, gsl::span<const int64_t> starts,
                    gsl::span<const int64_t> extents, gsl::span<const int64_t> steps)
      : is_string_tensor_(tensor.IsDataTypeString()),
        input_{reinterpret_cast<const byte*>(tensor.DataRaw())},
        element_size_(tensor.DataType()->Size()),
        extents_(extents),
        skips_(tensor_shape, extents, steps),
        indices_(extents.size(), 0) {
    auto dims = tensor_shape.GetDims();
    Init(dims, starts, steps);
  }

  void AdvanceOverInnerExtent() {
    size_t axis = skips_.size() - 1;
    AdvanceOverExtent(axis);
  }

  void AdvanceOverExtent(size_t axis) {
    input_ += skips_[axis] * element_size_;
    while (axis-- && ++indices_[axis] == extents_[axis]) {
      indices_[axis] = 0;
      input_ += skips_[axis] * element_size_;
    }
  }

  void IncrementInnerDimension() {
    input_ += inner_step_ * element_size_;
    if (++inner_counter_ == inner_extent_) {
      inner_counter_ = 0;
      AdvanceOverInnerExtent();
    }
  }

  const void* cur_input() const {
    return input_;
  }

  // Assumes SolitaryInnerStep() == true
  void* CopyInnermostAxisSolitaryInnerStep(void* output) {
    byte* out_bytes = reinterpret_cast<byte*>(output);
    auto bytes_to_copy = inner_extent_ * element_size_;

    if (!is_string_tensor_) {
      memcpy(output, input_, onnxruntime::narrow<size_t>(bytes_to_copy));
    } else {
      const std::string* input = reinterpret_cast<const std::string*>(input_);
      std::string* out = reinterpret_cast<std::string*>(output);
      std::copy(input, input + inner_extent_, out);
    }

    input_ += bytes_to_copy;
    out_bytes += bytes_to_copy;
    AdvanceOverInnerExtent();

    return out_bytes;
  }

  void* CopyContiguousInnermostAxes(void* output) {
    byte* out_bytes = nullptr;
    const auto bytes_to_copy = max_copying_elements_block_ * element_size_;
    if (SolitaryInnerStep()) {
      if (!is_string_tensor_) {
        memcpy(output, input_, onnxruntime::narrow<size_t>(bytes_to_copy));
      } else {
        const std::string* input = reinterpret_cast<const std::string*>(input_);
        std::string* out = reinterpret_cast<std::string*>(output);
        std::copy(input, input + max_copying_elements_block_, out);
      }
      input_ += bytes_to_copy;
      out_bytes = reinterpret_cast<byte*>(output);
      out_bytes += bytes_to_copy;
      AdvanceOverExtent(last_batching_axis_);
    } else {
      out_bytes = reinterpret_cast<byte*>(CopyInnermostAxisNonSolitaryInnerStep(output));
    }
    return out_bytes;
  }

  // Assumes generic inner_step_
  void* CopyInnermostAxisNonSolitaryInnerStep(void* output) {
    // need to special case std::string so the copy works correctly
    if (!is_string_tensor_) {
      // switch on element size so copy is efficient
      switch (element_size_) {
        case sizeof(uint8_t):
          output = TypedCopyInnermostAxisNonSolitaryInnerStep<uint8_t>(output);
          break;
        case sizeof(uint16_t):
          output = TypedCopyInnermostAxisNonSolitaryInnerStep<uint16_t>(output);
          break;
        case sizeof(uint32_t):
          output = TypedCopyInnermostAxisNonSolitaryInnerStep<uint32_t>(output);
          break;
        case sizeof(uint64_t):
          output = TypedCopyInnermostAxisNonSolitaryInnerStep<uint64_t>(output);
          break;
        default:
          ORT_THROW("Unexpected element size of ", element_size_);
      }
    } else {
      output = TypedCopyInnermostAxisNonSolitaryInnerStep<std::string>(output);
    }

    return output;
  }

 public:
  // splitting the function that copies the innermost dimension into 2 separate methods,
  // CopyInnermostAxisSolitaryInnerStep and CopyInnermostAxisNonSolitaryInnerStep,
  // as this is most likely being called within a loop
  // and we want to avoid the check inside to avoid overhead.
  // up to the caller to call the correct one based on SolitaryInnerStep().
  bool SolitaryInnerStep() const { return inner_step_ == 1; }

 private:
  template <typename T>
  void* TypedCopyInnermostAxisNonSolitaryInnerStep(void* output) {
    // sizeof(T) == element_size_
    T* out = reinterpret_cast<T*>(output);
    for (size_t i = 0; i < inner_extent_; ++i) {
      *out++ = *reinterpret_cast<const T*>(input_);
      IncrementInnerDimension();
    }

    return out;
  }

  const bool is_string_tensor_;
  // we do everything in this class using bytes to minimize binary size
  const byte* input_;
  const int64_t element_size_;

  gsl::span<const int64_t> extents_;
  size_t inner_counter_{}, inner_extent_;
  ptrdiff_t inner_step_;
  // Max copyable continuous block in elements
  int64_t max_copying_elements_block_;
  // The last slicing axis with step 1 that can combines continuous copyable blocks.
  // Used for skipping the max copyable block. If it is equal to the most inner dimension
  // then no extents can be combined for copying
  size_t last_batching_axis_;
  SliceSkips skips_;
  TensorShapeVector indices_;  // There is no index for innermost axis since it's a special case
};

// This provides easy sequential iteration over a subset of a tensor given a span of starts, extents & optionally steps
template <typename T>
struct SliceIterator : public SliceIteratorBase {
  SliceIterator(const Tensor& tensor, gsl::span<const int64_t> starts,
                gsl::span<const int64_t> extents, gsl::span<const int64_t> steps)
      : SliceIteratorBase(tensor, starts, extents, steps) {
  }

  // This construct takes a explicit tensor_shape which might be different from the shape defined in input tensor.
  // The explicit tensor_shape usually has inner most axis flattened. For example, given shape[1,4,4,2], if last axis
  // does not have padding or slice, then it will be flattened as [1,4,8] for better performance (One inner most copy instead of 4).
  // Also supports arbitrary positive and negative stepping along individual axes
  SliceIterator(const Tensor& tensor, const TensorShape& tensor_shape, gsl::span<const int64_t> starts,
                gsl::span<const int64_t> extents, gsl::span<const int64_t> steps)
      : SliceIteratorBase(tensor, tensor_shape, starts, extents, steps) {
  }

  // postfix iterator increment
  const T* operator++(int) {
    const T* input = static_cast<const T*>(cur_input());
    IncrementInnerDimension();
    return input;
  }

  // prefix iterator increment
  const T* operator++() {
    IncrementInnerDimension();
    return static_cast<const T*>(cur_input());
  }

  const T& operator*() const {
    return *static_cast<const T*>(cur_input());
  }

  // Assumes SolitaryInnerStep() == true
  T* CopyInnermostAxisSolitaryInnerStep(T* output) {
    void* new_output = SliceIteratorBase::CopyInnermostAxisSolitaryInnerStep(output);
    return static_cast<T*>(new_output);
  }

  // Assumes generic inner_step_
  T* CopyInnermostAxisNonSolitaryInnerStep(T* output) {
    void* new_output = SliceIteratorBase::CopyInnermostAxisNonSolitaryInnerStep(output);
    return static_cast<T*>(new_output);
  }

  T* CopyContiguousInnermostAxes(void* output) {
    void* new_output = SliceIteratorBase::CopyContiguousInnermostAxes(output);
    return static_cast<T*>(new_output);
  }
};

inline void CopyCpuTensor(const Tensor* src, Tensor* tgt) {
  void* target = tgt->MutableDataRaw();
  const void* source = src->DataRaw();

  if (target != source) {
    if (src->IsDataTypeString()) {
      auto src_span = src->DataAsSpan<std::string>();
      auto* dst_string = tgt->MutableData<std::string>();
      std::copy(src_span.begin(), src_span.end(), dst_string);
    } else {
      const auto element_size = src->DataType()->Size();
      const auto elements = src->Shape().Size();
      memcpy(target, source, SafeInt<size_t>(elements) * element_size);
    }
  }
}

// This provides easy sequential iteration over a subset of a tensor given a span of starts, extents & optionally steps
template <typename T>
struct WritableSliceIterator {
  WritableSliceIterator(Tensor& tensor, gsl::span<const int64_t> starts,
                        gsl::span<const int64_t> extents, gsl::span<const int64_t> steps)
      : input_(tensor.MutableData<T>()), extents_(extents), skips_(tensor.Shape(), extents, steps), indices_(extents.size(), 0) {
    auto dims = tensor.Shape().GetDims();
    Init(dims, starts, steps);
  }

  // This construct takes a explicit tensor_shape which might be different from the shape defined in input tensor.
  // The explicit tensor_shape usually has inner most axis flattened. For example, given shape[1,4,4,2], if last axis
  // does not have padding or slice, then it will be flattened as [1,4,8] for better performance (One inner most copy instead of 4).
  // Also supports arbitrary positive and negative stepping along individual axes
  WritableSliceIterator(Tensor& tensor, const TensorShape& tensor_shape, gsl::span<const int64_t> starts,
                        gsl::span<const int64_t> extents, gsl::span<const int64_t> steps)
      : input_(tensor.MutableData<T>()), extents_(extents), skips_(tensor_shape, extents, steps), indices_(extents.size(), 0) {
    auto dims = tensor_shape.GetDims();
    Init(dims, starts, steps);
  }

  // Initialize initial skip and inner_extent.
  void Init(gsl::span<const int64_t> dims, gsl::span<const int64_t> starts,
            gsl::span<const int64_t> steps) {
    ORT_ENFORCE(dims.size() == starts.size(),
                "dims.size()=", dims.size(), " != ", "starts.size()=", starts.size());

    ORT_ENFORCE(dims.size() == extents_.size(),
                "dims.size()=", dims.size(), " != ", "extents.size()=", extents_.size());

    ORT_ENFORCE(dims.size() == steps.size(),
                "dims.size()=", dims.size(), " != ", "steps.size()=", steps.size());

    SafeInt<size_t> pitch = 1;
    // Initial skip, so that input_ points to the first element to copy
    for (size_t i = dims.size(); i-- > 0;) {
      input_ += pitch * starts[i];
      pitch *= static_cast<size_t>(dims[i]);
    }

    inner_extent_ = onnxruntime::narrow<size_t>(extents_[SafeInt<size_t>(dims.size()) - 1]);
    inner_step_ = dims.size() == steps.size()
                      ? onnxruntime::narrow<size_t>(steps[SafeInt<size_t>(dims.size()) - 1])
                      : 1;
  }

  void AdvanceOverInnerExtent() {
    size_t axis = skips_.size() - 1;
    input_ += skips_[axis];
    while (axis-- && ++indices_[axis] == extents_[axis]) {
      indices_[axis] = 0;
      input_ += skips_[axis];
    }
  }

  void IncrementInnerDimension() {
    input_ += inner_step_;
    if (++inner_counter_ == inner_extent_) {
      inner_counter_ = 0;
      AdvanceOverInnerExtent();
    }
  }

  // postfix iterator increment
  const T* operator++(int) {
    const T* input = input_;
    IncrementInnerDimension();
    return input;
  }

  // prefix iterator increment
  const T* operator++() {
    IncrementInnerDimension();
    return input_;
  }

  const T& operator*() const {
    return *input_;
  }

  T& operator*() {
    return *input_;
  }

  bool SolitaryInnerStep() const { return inner_step_ == 1; }

  // spliting the function that copies the innermost dimension into 2 separate methods,
  // as this is most likely being called within a loop
  // and we want to avoid the check inside to avoid overhead
  // upto the caller to call the relevant one

  // Assumes inner_step_ == 1
  T* CopyInnermostAxisSolitaryInnerStep(T* output) {
    std::copy(input_, input_ + inner_extent_, output);
    input_ += inner_extent_;
    output += inner_extent_;
    AdvanceOverInnerExtent();
    return output;
  }

  T* CopyFromInnermostAxisSolitaryInnerStep(T* src) {
    std::copy(src, src + inner_extent_, input_);
    input_ += inner_extent_;
    src += inner_extent_;
    AdvanceOverInnerExtent();
    return src;
  }

  // Assumes generic inner_step_
  T* CopyInnermostAxisNonSolitaryInnerStep(T* output) {
    for (size_t i = 0; i < inner_extent_; ++i) {
      *output++ = *input_;
      input_ += inner_step_;
    }
    return output;
  }

  T* CopyFromInnermostAxisNonSolitaryInnerStep(T* src) {
    for (size_t i = 0; i < inner_extent_; ++i) {
      *input_ = *src++;
      IncrementInnerDimension();
    }
    return src;
  }

 private:
  T* input_;
  gsl::span<const int64_t> extents_;
  size_t inner_counter_{}, inner_extent_, inner_step_;
  SliceSkips skips_;
  TensorShapeVector indices_;  // There is no index for innermost axis since it's a special case
};

}  // namespace onnxruntime
