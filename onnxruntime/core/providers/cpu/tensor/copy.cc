//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/copy.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

std::vector<int64_t> StridesForTensor(const Tensor& tensor) {
  auto shape = tensor.Shape();
  auto strides = std::vector<int64_t>(shape.NumDimensions());
  int64_t running_size = 1;
  for (auto i = shape.NumDimensions(); i > 0; i--) {
    strides[i - 1] = running_size;
    running_size *= shape[i - 1];
  }

  return strides;
}

Status DispatchStridedCopy(concurrency::ThreadPool* thread_pool,
                           Tensor& dst,
                           std::ptrdiff_t dst_offset,
                           const std::vector<int64_t> dst_strides,
                           const TensorShape& copy_shape,
                           const Tensor& src,
                           const std::vector<int64_t> src_strides) {
  ORT_ENFORCE(dst.DataType() == src.DataType(), "src and dst types must match");

  // Manual dispatching: DispatchOnTensorType doesn't work here because we need to pass the type to the MutableData call
#define CALL_FOR_TYPE(T)                                                                                               \
  StridedCopy<T>(thread_pool, dst.MutableData<T>() + dst_offset, dst_strides, copy_shape, src.Data<T>(), src_strides); \
  break

  auto tensor_type = dst.DataType()->AsPrimitiveDataType()->GetDataType();
  switch (tensor_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      CALL_FOR_TYPE(float);
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      CALL_FOR_TYPE(bool);
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      CALL_FOR_TYPE(double);
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      CALL_FOR_TYPE(std::string);
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      CALL_FOR_TYPE(int8_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      CALL_FOR_TYPE(uint8_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      CALL_FOR_TYPE(int16_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      CALL_FOR_TYPE(uint16_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      CALL_FOR_TYPE(int32_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      CALL_FOR_TYPE(uint32_t);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      CALL_FOR_TYPE(int64_t);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      CALL_FOR_TYPE(uint64_t);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      CALL_FOR_TYPE(MLFloat16);
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      CALL_FOR_TYPE(BFloat16);
    default:
      ORT_ENFORCE(false, "Unknown tensor type of ", tensor_type);
  }
  return Status::OK();
}

namespace {

template <typename T>
inline void Copy1DNonContiguous(T* dst, int64_t dst_stride, const T* src, int64_t src_stride, std::ptrdiff_t count) {
  for (std::ptrdiff_t i = 0; i < count; i++) {
    dst[0] = src[0];
    dst += dst_stride;
    src += src_stride;
  }
}

template <typename T>
inline void Copy1DContiguous(T* dst, const T* src, std::ptrdiff_t count) {
  memcpy(dst, src, count * sizeof(T));
}
template <>
inline void Copy1DContiguous<std::string>(std::string* dst, const std::string* src, std::ptrdiff_t count) {
  Copy1DNonContiguous(dst, 1, src, 1, count);
}

template <typename T>
inline void Copy1D(T* dst, int64_t dst_stride, const T* src, int64_t src_stride, std::ptrdiff_t count) {
  if (dst_stride == 1 && src_stride == 1) {
    Copy1DContiguous(dst, src, count);
  } else {
    Copy1DNonContiguous(dst, dst_stride, src, src_stride, count);
  }
}

template <>
inline void Copy1D<std::string>(std::string* dst, int64_t dst_stride, const std::string* src, int64_t src_stride, std::ptrdiff_t count) {
  // strings should always be copied using the for loop
  Copy1DNonContiguous(dst, dst_stride, src, src_stride, count);
}

struct NdCounter {
  NdCounter(const std::vector<int64_t>& shape, std::ptrdiff_t first, std::ptrdiff_t last)
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
    std::ptrdiff_t span_end = std::min<std::ptrdiff_t>(last, current_offset + elements_in_dimension);
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
  std::vector<int64_t> current_index;
  const std::vector<int64_t>& shape;
};

/*
    Check if we can coalesce dim with dim + 1.

    We can do this if:
      * either of the dims have shape 1
      * strides[dim + 1] * shape[dim + 1] = strides[dim] (for all tensors)
*/
inline bool CanCoalesce(
    std::initializer_list<std::reference_wrapper<std::vector<int64_t>>>& tensors_strides,
    const std::vector<int64_t>& shape,
    std::size_t dim,
    std::size_t ndim) {
  auto shape_dim = shape[dim];
  auto shape_ndim = shape[ndim];
  if (shape_dim == 1 || shape_ndim == 1) {
    return true;
  }

  for (const auto& strides_ : tensors_strides) {
    std::vector<int64_t>& strides = strides_.get();
    if (shape_ndim * strides[ndim] != strides[dim]) {
      return false;
    }
  }
  return true;
}

/*
    Copy the stride from ndim to dim in all tensors.
*/
inline void CopyStride(
    std::initializer_list<std::reference_wrapper<std::vector<int64_t>>>& tensors_strides,
    std::size_t dim, std::size_t ndim) {
  for (const auto& strides_ : tensors_strides) {
    std::vector<int64_t>& strides = strides_.get();
    strides[dim] = strides[ndim];
  }
}

}  // namespace

/*
    Coalesce contiguous dimensions in the tensors. Operates inplace on the function arguments.
*/
void CoalesceDimensions(std::initializer_list<std::reference_wrapper<std::vector<int64_t>>>&& tensors_strides, std::vector<int64_t>& shape) {
  const std::size_t dims = shape.size();

  // the current dimension is the one we are attempting to "coalesce onto"
  std::size_t current_dim = 0;

  for (std::size_t dim = 1; dim < dims; dim++) {
    // check if dim can be coalesced with current_dim
    if (CanCoalesce(tensors_strides, shape, current_dim, dim)) {
      if (shape[dim] != 1) {
        CopyStride(tensors_strides, current_dim, dim);
      }
      shape[current_dim] *= shape[dim];
    } else {
      current_dim++;

      if (current_dim != dim) {
        // we have coaleseced at least one value before this: bump forward the values into the correct place
        CopyStride(tensors_strides, current_dim, dim);
        shape[current_dim] = shape[dim];
      }
    }
  }

  shape.resize(current_dim + 1);
  for (const auto& strides_ : tensors_strides) {
    std::vector<int64_t>& strides = strides_.get();
    strides.resize(current_dim + 1);
  }
}

template <typename T>
void StridedCopy(concurrency::ThreadPool* thread_pool,
                 T* dst,
                 const std::vector<int64_t>& dst_strides_,
                 const TensorShape& copy_shape_,
                 const T* src,
                 const std::vector<int64_t>& src_strides_) {
  // Coalesce dimensions
  std::vector<int64_t> dst_strides = dst_strides_;
  std::vector<int64_t> src_strides = src_strides_;
  std::vector<int64_t> copy_shape(copy_shape_.GetDims());

  CoalesceDimensions({dst_strides, src_strides}, copy_shape);
  ORT_ENFORCE(dst_strides.size() == src_strides.size() && src_strides.size() == copy_shape.size(), "src and dst must have same shape");

  const std::size_t dims = copy_shape.size();
  // We will iterate over the output dimensions
  int64_t num_iterations = 1;
  for (std::size_t dim = 0; dim < dims; dim++) {
    num_iterations *= copy_shape[dim];
  }

  if (num_iterations <= 1) {
    // scalar edge case
    dst[0] = src[0];
    return;
  }

  // TODOs for when we have strided tensors:
  // - Reorder dimensions so that we iterate along the smallest strides first

  ORT_ENFORCE(dims > 0);

  if (dims <= 2 && src_strides[dims - 1] == 1 && dst_strides[dims - 1] == 1) {
    // Fast path for 2D copies that skips the NdCounter required in the general case.
    // This avoids overhead which becomes noticable at smaller iteration sizes.
    //
    // After coalescing, the case is actually quite common since all tensors in ORT are contiguous

    int64_t dst_stride = dims == 2 ? dst_strides[0] : 0;
    int64_t src_stride = dims == 2 ? src_strides[0] : 0;

    // the size of contiguous spans that we can copy before having to advance the non-contiguous stride
    int64_t contiguous_span_size = dims == 2 ? copy_shape[1] : copy_shape[0];

    concurrency::ThreadPool::TryParallelFor(
        thread_pool, num_iterations,
        {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), 1.0F},
        [src_stride, dst_stride, dst, src, contiguous_span_size](std::ptrdiff_t first, std::ptrdiff_t last) {
          // get the current inner and outer index
          int64_t inner = first % contiguous_span_size;
          int64_t outer = first / contiguous_span_size;

          std::ptrdiff_t dst_idx = outer * dst_stride + inner;
          std::ptrdiff_t src_idx = outer * src_stride + inner;

          // Step 1: if there is anything left in the contiguous span that we are starting in, finish copying it
          if (inner != 0) {
            auto elements_to_copy = contiguous_span_size - inner;
            // never copy more than what is in our partition
            elements_to_copy = std::min<std::ptrdiff_t>(elements_to_copy, last - first);
            Copy1DContiguous<T>(dst + dst_idx, src + src_idx, elements_to_copy);
            inner = 0;
            outer++;
            first += elements_to_copy;

            // reset the dst and src idx now that we are aligned to the start of a contiguous span
            dst_idx = outer * dst_stride;
            src_idx = outer * src_stride;
          }

          // Step 2: copy contiguous span by contiguous span until we reach the penultimate span
          while (first < last - contiguous_span_size) {
            Copy1DContiguous<T>(dst + dst_idx, src + src_idx, contiguous_span_size);
            dst_idx += dst_stride;
            src_idx += src_stride;
            first += contiguous_span_size;
          }
          // Step 3: finish off the last (possibly partial) span manually, making sure that we don't go past the last
          // element in our partition
          ORT_ENFORCE(last >= first);
          auto last_span_size = last - first;
          Copy1DContiguous<T>(dst + dst_idx, src + src_idx, last_span_size);
        });
  } else {
    concurrency::ThreadPool::TryParallelFor(
        thread_pool, num_iterations,
        {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), 1.0F},
        [copy_shape, dst_strides, dst, src, src_strides, dims](std::ptrdiff_t first, std::ptrdiff_t last) {
          NdCounter counter(copy_shape, first, last);

          auto last_dst_stride = dst_strides[dims - 1];
          auto last_src_stride = src_strides[dims - 1];

          auto iter_size = counter.NextStepSize();
          while (iter_size > 0) {
            // Compute the src and dst addresses
            std::ptrdiff_t dst_idx = 0;
            std::ptrdiff_t src_idx = 0;
            for (std::size_t dim = 0; dim < dims; dim++) {
              dst_idx += counter.current_index[dim] * dst_strides[dim];
              src_idx += counter.current_index[dim] * src_strides[dim];
            }
            // we can copy until the current dimension is done (or until we hit the last element we are trying to copy)
            Copy1D<T>(dst + dst_idx, last_dst_stride, src + src_idx, last_src_stride, iter_size);

            counter.Step(iter_size);
            iter_size = counter.NextStepSize();
          }
          ORT_ENFORCE(counter.current_offset == last);
        });
  }
}
}  // namespace onnxruntime
