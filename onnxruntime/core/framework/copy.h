// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/op_kernel_type_control_utils.h"

#include <vector>

namespace onnxruntime {

void CoalesceDimensions(
    std::initializer_list<std::reference_wrapper<TensorShapeVector>>&& tensors_strides, TensorShapeVector& shape);

TensorShapeVector StridesForTensor(const Tensor& tensor);

namespace strided_copy_detail {

template <typename T>
void Copy1DNonContiguous(T* dst, int64_t dst_stride, const T* src, int64_t src_stride, std::ptrdiff_t count) {
  for (std::ptrdiff_t i = 0; i < count; i++) {
    dst[0] = src[0];
    dst += dst_stride;
    src += src_stride;
  }
}

template <typename T>
void Copy1DContiguous(T* dst, const T* src, std::ptrdiff_t count) {
  if constexpr (std::is_same_v<std::string, T>) {
    Copy1DNonContiguous(dst, 1, src, 1, count);
  } else {
    memcpy(dst, src, count * sizeof(T));
  }
}

template <typename T>
void Copy1D(T* dst, int64_t dst_stride, const T* src, int64_t src_stride, std::ptrdiff_t count) {
  if constexpr (std::is_same_v<std::string, T>) {
    // strings should always be copied using the for loop
    Copy1DNonContiguous(dst, dst_stride, src, src_stride, count);
  } else {
    if (dst_stride == 1 && src_stride == 1) {
      Copy1DContiguous(dst, src, count);
    } else {
      Copy1DNonContiguous(dst, dst_stride, src, src_stride, count);
    }
  }
}

struct NdCounter {
  NdCounter(const TensorShapeVector& shape, std::ptrdiff_t first, std::ptrdiff_t last)
      : dims(shape.size()),
        last_dim_size(shape[dims - 1]),
        current_offset(first),
        last(last),
        current_index(dims),
        shape(shape) {
    // compute the initial n-dimensional index
    int64_t remaining_index = first;
    // Iterate from dims to 1 so we don't roll over to positive on the bounds check
    for (std::size_t dim = dims; dim > 0; dim--) {
      auto shape_val = shape[dim - 1];
      current_index[dim - 1] = remaining_index % shape_val;
      remaining_index /= shape_val;
    }
  }

  /*
      Return the size of the largest step in the last dimension.
  */
  std::ptrdiff_t NextStepSize() const {
    auto elements_in_dimension = last_dim_size - current_index[dims - 1];
    std::ptrdiff_t span_end =
        std::min<std::ptrdiff_t>(last, current_offset + static_cast<std::ptrdiff_t>(elements_in_dimension));
    return span_end - current_offset;
  }

  /*
      Advance the counter by step_size elements.
  */
  void Step(std::ptrdiff_t step_size) {
    current_offset += step_size;
    current_index[dims - 1] += step_size;

    // update the current_nd_idx if needed
    std::size_t dim = dims - 1;
    while (dim > 0 && current_index[dim] >= shape[dim]) {
      current_index[dim] = 0;
      dim--;
      current_index[dim]++;
    }
  }

  const std::size_t dims;
  const int64_t last_dim_size;
  ptrdiff_t current_offset;
  const ptrdiff_t last;
  TensorShapeVector current_index;
  const TensorShapeVector& shape;
};
}  // namespace strided_copy_detail

template <typename T>
void StridedCopy(concurrency::ThreadPool* thread_pool,
                 T* dst,
                 const TensorShapeVector& dst_strides_in,
                 const TensorShape& copy_shape_in,
                 const T* src,
                 const TensorShapeVector& src_strides_in) {
  // Coalesce dimensions
  TensorShapeVector dst_strides(dst_strides_in);
  TensorShapeVector src_strides(src_strides_in);
  TensorShapeVector copy_shape = copy_shape_in.AsShapeVector();

  CoalesceDimensions({dst_strides, src_strides}, copy_shape);
  ORT_ENFORCE(dst_strides.size() == src_strides.size() &&
                  src_strides.size() == copy_shape.size() &&
                  !copy_shape.empty(),
              "src and dst must have same shape and not be rank 0.");

  const int64_t total_num_elements_to_copy = copy_shape_in.Size();

  ORT_ENFORCE(total_num_elements_to_copy >= 0, "copy shape must have non-negative size");

  if (total_num_elements_to_copy == 0) {
    // empty edge case
    return;
  }

  if (total_num_elements_to_copy == 1) {
    // scalar edge case
    dst[0] = src[0];
    return;
  }

  const std::size_t dims = copy_shape.size();

  // TODOs for when we have strided tensors:
  // - Reorder dimensions so that we iterate along the smallest strides first

  if (dims <= 2 && src_strides[dims - 1] == 1 && dst_strides[dims - 1] == 1) {
    // Fast path for 2D copies that skips the NdCounter required in the general case.
    // This avoids overhead which becomes noticeable at smaller iteration sizes.
    //
    // After coalescing, the case is actually quite common since all tensors in ORT are contiguous

    std::ptrdiff_t dst_stride = static_cast<std::ptrdiff_t>(dims == 2 ? dst_strides[0] : 0);
    std::ptrdiff_t src_stride = static_cast<std::ptrdiff_t>(dims == 2 ? src_strides[0] : 0);

    // the size of contiguous spans that we can copy before having to advance the non-contiguous stride
    std::ptrdiff_t contiguous_span_size = static_cast<std::ptrdiff_t>(dims == 2 ? copy_shape[1] : copy_shape[0]);

    concurrency::ThreadPool::TryParallelFor(
        thread_pool, static_cast<std::ptrdiff_t>(total_num_elements_to_copy),
        {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), 1.0F},
        [src_stride, dst_stride, dst, src, contiguous_span_size](std::ptrdiff_t first, std::ptrdiff_t last) {
          // get the current inner and outer index
          std::ptrdiff_t inner = first % contiguous_span_size;
          std::ptrdiff_t outer = first / contiguous_span_size;

          std::ptrdiff_t dst_idx = outer * dst_stride + inner;
          std::ptrdiff_t src_idx = outer * src_stride + inner;

          // Step 1: if there is anything left in the contiguous span that we are starting in, finish copying it
          if (inner != 0) {
            auto elements_to_copy = contiguous_span_size - inner;
            // never copy more than what is in our partition
            elements_to_copy = std::min<std::ptrdiff_t>(elements_to_copy, last - first);
            strided_copy_detail::Copy1DContiguous<T>(dst + dst_idx, src + src_idx, elements_to_copy);
            inner = 0;
            outer++;
            first += elements_to_copy;

            // reset the dst and src idx now that we are aligned to the start of a contiguous span
            dst_idx = outer * dst_stride;
            src_idx = outer * src_stride;
          }

          // Step 2: copy contiguous span by contiguous span until we reach the penultimate span
          while (first < last - contiguous_span_size) {
            strided_copy_detail::Copy1DContiguous<T>(dst + dst_idx, src + src_idx, contiguous_span_size);
            dst_idx += dst_stride;
            src_idx += src_stride;
            first += contiguous_span_size;
          }
          // Step 3: finish off the last (possibly partial) span manually, making sure that we don't go past the last
          // element in our partition
          ORT_ENFORCE(last >= first);
          auto last_span_size = last - first;
          strided_copy_detail::Copy1DContiguous<T>(dst + dst_idx, src + src_idx, last_span_size);
        });
  } else {
    // enforce that the lambda doesn't change anything
    const TensorShapeVector& const_dst_strides = dst_strides;
    const TensorShapeVector& const_src_strides = src_strides;
    const TensorShapeVector& const_copy_shape = copy_shape;

    concurrency::ThreadPool::TryParallelFor(
        thread_pool, static_cast<std::ptrdiff_t>(total_num_elements_to_copy),
        {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), 1.0F},
        [&const_copy_shape, &const_dst_strides, dst, src, &const_src_strides, dims](std::ptrdiff_t first,
                                                                                    std::ptrdiff_t last) {
          strided_copy_detail::NdCounter counter(const_copy_shape, first, last);

          auto last_dst_stride = const_dst_strides[dims - 1];
          auto last_src_stride = const_src_strides[dims - 1];

          auto iter_size = counter.NextStepSize();
          while (iter_size > 0) {
            // Compute the src and dst addresses
            std::ptrdiff_t dst_idx = 0;
            std::ptrdiff_t src_idx = 0;
            for (std::size_t dim = 0; dim < dims; dim++) {
              dst_idx += static_cast<std::ptrdiff_t>(counter.current_index[dim] * const_dst_strides[dim]);
              src_idx += static_cast<std::ptrdiff_t>(counter.current_index[dim] * const_src_strides[dim]);
            }
            // we can copy until the current dimension is done (or until we hit the last element we are trying to copy)
            strided_copy_detail::Copy1D<T>(dst + dst_idx, last_dst_stride, src + src_idx, last_src_stride, iter_size);

            counter.Step(iter_size);
            iter_size = counter.NextStepSize();
          }
          ORT_ENFORCE(counter.current_offset == last);
        });
  }
}

// call StridedCopy if there is a type with the same size as T in the set of EnabledTypes
// e.g. if uint32_t is enabled all 4 byte types are supported
template <typename EnabledTypes, typename T>
inline bool StridedCopyIfEnabled(concurrency::ThreadPool* thread_pool,
                                 Tensor& dst,
                                 std::ptrdiff_t dst_offset,
                                 const TensorShapeVector& dst_strides,
                                 const TensorShape& copy_shape,
                                 const Tensor& src,
                                 std::ptrdiff_t src_offset,
                                 const TensorShapeVector& src_strides) {
  constexpr bool enabled = utils::HasTypeWithSameSize<EnabledTypes, T>();
  if constexpr (enabled) {
    // T doesn't necessarily match the data type in src or dst so use reinterpret_cast.
    // it will be a type with the same size though, which is all that matters given we're only copying bits.
    StridedCopy<T>(thread_pool,
                   reinterpret_cast<T*>(dst.MutableDataRaw()) + dst_offset,
                   dst_strides, copy_shape,
                   reinterpret_cast<const T*>(src.DataRaw()) + src_offset,
                   src_strides);
  }

  return enabled;
}

// EnabledTypes is an onnxruntime::TypeList with the enabled types in this build.
// see "core/framework/element_type_lists.h" for default lists or the usage in
// onnxruntime/core/providers/cpu/tensor/concat.cc
template <typename EnabledDataTypes>
Status DispatchStridedCopy(concurrency::ThreadPool* thread_pool,
                           Tensor& dst,
                           std::ptrdiff_t dst_offset,
                           const TensorShapeVector& dst_strides,
                           const TensorShape& copy_shape,
                           const Tensor& src,
                           std::ptrdiff_t src_offset,
                           const TensorShapeVector& src_strides) {
  ORT_ENFORCE(dst.DataType() == src.DataType(), "src and dst types must match");

  bool supported = false;
  if (src.IsDataTypeString()) {
    if constexpr (utils::HasType<EnabledDataTypes, std::string>()) {
      supported = true;
      StridedCopy(thread_pool, dst.MutableData<std::string>() + dst_offset, dst_strides, copy_shape,
                  src.Data<std::string>() + src_offset, src_strides);
    }
  } else {
    const auto element_size = src.DataType()->Size();
    switch (element_size) {
      case sizeof(uint32_t):
        supported = StridedCopyIfEnabled<EnabledDataTypes, uint32_t>(thread_pool, dst, dst_offset, dst_strides,
                                                                     copy_shape, src, src_offset, src_strides);
        break;
      case sizeof(uint64_t):
        supported = StridedCopyIfEnabled<EnabledDataTypes, uint64_t>(thread_pool, dst, dst_offset, dst_strides,
                                                                     copy_shape, src, src_offset, src_strides);
        break;
      case sizeof(uint16_t):
        supported = StridedCopyIfEnabled<EnabledDataTypes, uint16_t>(thread_pool, dst, dst_offset, dst_strides,
                                                                     copy_shape, src, src_offset, src_strides);
        break;
      case sizeof(uint8_t):
        static_assert(sizeof(bool) == sizeof(uint8_t), "Need to enable separate case for 'bool' on this platform.");
        supported = StridedCopyIfEnabled<EnabledDataTypes, uint8_t>(thread_pool, dst, dst_offset, dst_strides,
                                                                    copy_shape, src, src_offset, src_strides);
        break;
      // It's possible that bool is not 1 byte. static_assert above checks if we need to enable this on a platform.
      // case sizeof(bool):
      //  supported = StridedCopyIfEnabled<EnabledDataTypes, bool>(thread_pool, dst, dst_offset, dst_strides,
      //                                                           copy_shape, src, src_offset, src_strides);
      //  break;
      default:
        // leave 'supported' as false
        break;
    }
  }

  return !supported ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported input data type of ", src.DataType())
                    : Status::OK();
}

}  // namespace onnxruntime
