// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stddef.h>
#include <iostream>
#include <string>
#include <vector>

#include "gsl/gsl"
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor_shape.h"
#include "onnxruntime_config.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"

namespace onnxruntime {
// TODO: Do we need this class or is IAllocator::MakeUniquePtr sufficient/better
class BufferDeleter {
 public:
  BufferDeleter() : alloc_(nullptr) {}
  BufferDeleter(AllocatorPtr alloc)
      : alloc_(alloc) {}

  void operator()(void* p) const {
    if (alloc_)
      alloc_->Free(p);
  }

 private:
  // TODO: we may need consider the lifetime of alloc carefully
  // The alloc_ here is the allocator that used to allocate the buffer
  // And need go with the unique_ptr together. If it is using our internal
  // allocator, it is ok as our allocators are global managed. But if it
  // is provide by user, user need to be very careful about it.
  // A weak_ptr may be a choice to reduce the impact, but that require to
  // change our current allocator mgr to use shared_ptr. Will revisit it
  // later.
  AllocatorPtr alloc_;
};

using BufferUniquePtr = std::unique_ptr<void, BufferDeleter>;
using BufferNakedPtr = void*;
//TODO:ensure dtype_!=nullptr
#ifdef __GNUC__
#pragma GCC diagnostic push
#ifdef HAS_NULL_DEREFERENCE
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
#endif
/*
  We want to keep tensor as simple as possible, it is just a placeholder
  for a piece of memory, with additional shape information.
  Memory is owned and managed by Executor / Workspace, so Tensor just uses
  it, and won't do any allocation / release.
*/
class Tensor final {
 public:
  Tensor() = default;  // to allow creating vector<Tensor> to support seq(tensor)

  /**
   * Create tensor with given type, shape, pre-allocate memory and allocator info.
   * This function won't check if the preallocated buffer(p_data) has enough room for the shape.
   * \param data A preallocated buffer. Can be NULL if the shape is empty.
   *              Tensor does not own the data and will not delete it
   * \param alloc Where the buffer('data') was allocated from
   */
  Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc,
         int64_t offset = 0);

  /**
   * Deprecated. The orginal design is this Tensor class won't do any allocation / release.
   * However, this function will allocate the buffer for the shape, and do placement new if p_type is string tensor.
   */
  Tensor(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator, int64_t offset = 0);

  ~Tensor();

  //Move is allowed
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Tensor);

  Tensor(Tensor&& other) noexcept;

  Tensor& operator=(Tensor&& other) noexcept;

  /**
     Returns the data type.
  */
  MLDataType DataType() const { return dtype_; }

  /**
     Returns the data type enum contant
  */
  int32_t GetElementType() const {
    return dtype_->GetDataType();
  }

  // Check if contains string data. This is a separate
  // interface bc it is frequently used.
  bool IsDataTypeString() const {
    return utils::IsPrimitiveDataType<std::string>(dtype_);
  }

  // Checks if the Tensor contains data type T
  template<class T>
  bool IsDataType () const {
    return utils::IsPrimitiveDataType<T>(dtype_);
  }

  /**
     Returns the shape of the tensor.
  */
  const TensorShape& Shape() const noexcept { return shape_; }

  /**
     Returns the location of the tensor's memory
  */
  const OrtMemoryInfo& Location() const { return alloc_info_; }

  /**
     May return nullptr if tensor size is zero
  */
  template <typename T>
  T* MutableData() {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    return reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset_);
  }

  /**
     May return nullptr if tensor size is zero
  */
  template <typename T>
  gsl::span<T> MutableDataAsSpan() {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    T* data = reinterpret_cast<T*>(static_cast<char*>(p_data_) + byte_offset_);
    return gsl::make_span(data, shape_.Size());
  }

  template <typename T>
  const T* Data() const {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    return reinterpret_cast<const T*>(static_cast<char*>(p_data_) + byte_offset_);
  }

  template <typename T>
  gsl::span<const T> DataAsSpan() const {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    const T* data = reinterpret_cast<const T*>(static_cast<char*>(p_data_) + byte_offset_);
    return gsl::make_span(data, shape_.Size());
  }

  void* MutableDataRaw(MLDataType type) {
    ORT_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return p_data_;
  }

  const void* DataRaw(MLDataType type) const {
    ORT_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return p_data_;
  }

  void* MutableDataRaw() noexcept {
    return p_data_;
  }

  const void* DataRaw() const noexcept {
    return p_data_;
  }

  /**
   * Resizes the tensor without touching underlying storage.
   * This requires the total size of the tensor to remains constant.
   * @warning this function is NOT thread-safe.
   */
  inline void Reshape(const TensorShape& new_shape) {
    ORT_ENFORCE(shape_.Size() == new_shape.Size(),
                "Tensor size (" + std::to_string(shape_.Size()) +
                    ") != new size (" + std::to_string(new_shape.Size()) + ")");
    shape_ = new_shape;
  }

  /**
  The number of bytes of data.
  */
  size_t SizeInBytes() const;

  // More API methods.
 private:
  void Init(MLDataType p_type,
            const TensorShape& shape,
            void* p_raw_data,
            AllocatorPtr deleter,
            int64_t offset = 0);

  void ReleaseBuffer();

  void* p_data_;
  /**
     if buffer_deleter_ is null, it means tensor does not own the buffer.
     otherwise tensor will use the deleter to release the buffer when
     tensor is released.
  */
  AllocatorPtr buffer_deleter_;

  TensorShape shape_;
  const PrimitiveDataTypeBase* dtype_;
  OrtMemoryInfo alloc_info_;
  int64_t byte_offset_;
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}  // namespace onnxruntime
