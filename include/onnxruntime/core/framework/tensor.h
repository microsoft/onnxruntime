// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stddef.h>
#include <iostream>
#include <string>
#include <vector>

#include "core/common/gsl.h"
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/buffer_deleter.h"
#include "onnxruntime_config.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/storage.h"

struct OrtValue;

namespace onnxruntime {
using ShardDims = TensorShapeVector;
// TODO:ensure dtype_!=nullptr
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
  // NB! Removing Create() methods returning unique_ptr<Tensor>. Still available in other EPs that are dynamically linked.
  // Strive not to allocate Tensor with new/delete as it is a shallow class and using it by value is just fine.
  // Use InitOrtValue() methods to allocate for OrtValue.

  Tensor() = default;  // to allow creating vector<Tensor> to support seq(tensor)

  /**
   * Create tensor with given type, shape, pre-allocated memory and allocator info.
   * This function won't check if the preallocated buffer(p_data) has enough room for the shape.
   * \param p_type Data type of the tensor
   * \param shape Shape of the tensor
   * \param p_data A preallocated buffer. Can be NULL if the shape is empty.
   *              Tensor does not own the data and will not delete it
   * \param alloc Where the buffer('p_data') was allocated from
   * \param offset Offset in bytes to start of Tensor within p_data.
   * \param strides Strides span. Can be empty if the tensor is contiguous.
   */
  Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc,
         ptrdiff_t offset = 0, gsl::span<const int64_t> strides = {});

  Tensor(MLDataType p_type, const TensorShape& shape, std::vector<std::shared_ptr<IAllocator>> allocators,
               gsl::span<const int64_t> strides={});

  static void InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::vector<std::shared_ptr<IAllocator>> allocators,
                          OrtValue& ort_value, gsl::span<const int64_t> strides={});
  /// <summary>
  /// Creates an instance of Tensor on the heap using the appropriate __ctor and
  /// initializes OrtValue with it.
  /// </summary>
  /// <param name="p_type"></param>
  /// <param name="shape"></param>
  /// <param name="p_data"></param>
  /// <param name="info"></param>
  /// <param name="offset"></param>
  /// <param name="strides"></param>
  static void InitOrtValue(MLDataType p_type, const TensorShape& shape,
                           void* p_data, const OrtMemoryInfo& location,
                           OrtValue& ort_value, ptrdiff_t offset = 0,
                           gsl::span<const int64_t> strides = {});

  /// <summary>
  /// Creates an instance of Tensor who own the pre-allocated buffer.
  /// </summary>
  /// <param name="p_type"></param>
  /// <param name="shape"></param>
  /// <param name="p_data"></param>
  /// <param name="allocator"></param>
  /// <param name="offset"></param>
  /// <param name="strides"></param>
  static void InitOrtValue(MLDataType p_type, const TensorShape& shape,
                           void* p_data, std::shared_ptr<IAllocator> allocator,
                           OrtValue& ort_value, ptrdiff_t offset = 0,
                           gsl::span<const int64_t> strides = {});

  static size_t CalculateTensorStorageSize(MLDataType p_type,
                                           const TensorShape& shape,
                                           gsl::span<const int64_t> strides = {});

  /**
   * Deprecated. The original design is this Tensor class won't do any allocation / release.
   * However, this function will allocate the buffer for the shape, and do placement new if p_type is string tensor.
   */
  Tensor(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator,
         gsl::span<const int64_t> strides = {});

  /// <summary>
  /// Creates an instance of Tensor on the heap using the appropriate __ctor and
  /// initializes OrtValue with it.
  /// </summary>
  /// <param name="elt_type"></param>
  /// <param name="shape"></param>
  /// <param name="allocator"></param>
  /// <param name="ort_value"></param>
  /// <param name="strides"></param>
  static void InitOrtValue(MLDataType elt_type,
                           const TensorShape& shape,
                           std::shared_ptr<IAllocator> allocator,
                           OrtValue& ort_value,
                           gsl::span<const int64_t> strides = {});

  /// <summary>
  /// Creates an instance of Tensor on the heap using the appropriate __ctor and
  /// initializes OrtValue with it.
  /// </summary>
  /// <param name="tensor"></param>
  /// <param name="ort_value"></param>
  static void InitOrtValue(Tensor&& tensor, OrtValue& ort_value);

  /**
   * Create tensor with given type, shape, pre-allocated memory and allocator which will be used to free the pre-allocated memory.
   * This function won't check if the preallocated buffer(p_data) has enough room for the shape.
   * However, this function will de-allocate the buffer upon the tensor getting destructed.
   * \param p_type Data type of the tensor
   * \param shape Shape of the tensor
   * \param p_data A preallocated buffer. Can be NULL if the shape is empty.
   *              Tensor will own the memory and will delete it when the tensor instance is destructed.
   * \param deleter Allocator used to free the pre-allocated memory
   * \param offset Offset in bytes to start of Tensor within p_data.
   * \param strides Strides span. Can be empty if the tensor is contiguous.
   */
  Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, std::shared_ptr<IAllocator> deleter,
         ptrdiff_t offset = 0, gsl::span<const int64_t> strides = {});

  ~Tensor();

  // Move is allowed
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(Tensor);

  Tensor(Tensor&& other) noexcept;

  Tensor& operator=(Tensor&& other) noexcept;

  /**
     Returns the data type.
  */
  MLDataType DataType() const { return dtype_; }

  /**
     Returns the data type enum constant
     @remarks Use utils::ToTensorProtoElementType<T> for comparison.
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
  template <class T>
  bool IsDataType() const {
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
    return reinterpret_cast<T*>(storage_->Data(0));
  }

  /**
     May return nullptr if tensor size is zero
  */
  template <typename T>
  gsl::span<T> MutableDataAsSpan() {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    T* data = reinterpret_cast<T*>(storage_->Data(0));
    return gsl::make_span(data, static_cast<size_t>(shape_.Size()));
  }

  template <typename T>
  const T* Data() const {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    return reinterpret_cast<const T*>(storage_->Data(0));
  }

  template <typename T>
  gsl::span<const T> DataAsSpan() const {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    const T* data = reinterpret_cast<const T*>(storage_->Data(0));
    using TSize = typename gsl::span<T>::size_type;
    auto numElems = static_cast<TSize>(shape_.Size());
    return gsl::make_span(data, numElems);
  }

  void* MutableDataRaw(MLDataType type) {
    ORT_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return storage_->Data(0);
  }

  const void* DataRaw(MLDataType type) const {
    ORT_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return storage_->Data(0);
  }

  void* MutableDataRaw() noexcept {
    return storage_->Data(0);
  }

  const void* DataRaw() const noexcept {
    return storage_->Data(0);
  }

  bool OwnsBuffer() const noexcept {
    return storage_->OwnsBuffer();
  }

  /** Accessors for shards */
  template <typename T>
  T* MutableData(size_t offset) {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    return reinterpret_cast<T*>(storage_->Data(offset));
  }

  template <typename T>
  gsl::span<T> MutableDataAsSpan(size_t offset) {
    T* data = MutableData<T>(offset);
    auto numElems = static_cast<size_t>(storage_->Size(offset) / dtype_->Size());
    return gsl::make_span(data, static_cast<size_t>(numElems));
  }

  template <typename T>
  const T* Data(size_t offset) const {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    return reinterpret_cast<const T*>(storage_->Data(offset));
  }

  template <typename T>
  gsl::span<const T> DataAsSpan(size_t offset) const {
    // Type check
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Tensor type mismatch. ",
                "T ", "!=", dtype_);
    const T* data = reinterpret_cast<const T*>(storage_->Data(0));
    using TSize = typename gsl::span<T>::size_type;
    auto numElems = static_cast<TSize>(storage_->Size(offset) / dtype_->Size());
    return gsl::make_span(data, numElems);
  }

  void* MutableDataRaw(MLDataType type, size_t offset) {
    ORT_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return storage_->Data(offset);
  }

  const void* DataRaw(MLDataType type, size_t offset) const {
    ORT_ENFORCE(type == dtype_, "Tensor type mismatch.", type, "!=", dtype_);
    return storage_->Data(offset);
  }

  void* MutableDataRaw(size_t index) noexcept { return storage_->Data(index); }
  const void* DataRaw(size_t index) const noexcept { return storage_->Data(index); }
  /** End of accessors for shards */

  //
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
   * Get the byte offset with respect to the p_data
   * @warning this is a temporary solution for reusing the buffer bigger than needed.
   * @warning use with caution - make sure you do boundary check before calling this method (see view.cc)
   */
  inline ptrdiff_t ByteOffset() const {
    return storage_->Offset(0);
  }

  /**
   * Set the byte offset with respect to the p_data
   * @warning this is a temporary solution for reusing the buffer bigger than needed.
   */
  inline void SetByteOffset(ptrdiff_t byte_offset) {
    storage_->SetOffset(byte_offset);
  }

  /**
  The number of bytes of data.
  */
  size_t SizeInBytes() const;

#ifdef ENABLE_STRIDED_TENSORS
  /**
   * Get the strides of the tensor.
   */
  gsl::span<const int64_t> Strides() const;

  /**
   * Return if the tensor is contiguous.
   */
  bool IsContiguous() const noexcept { return is_contiguous_; }

  /**
   * Set strides.
   */
  void SetShapeAndStrides(const TensorShape& new_shape, gsl::span<const int64_t> new_strides);
#endif

  // More API methods.
 private:
  void Init(MLDataType p_type,
            std::shared_ptr<Storage> storage,
            const TensorShape& shape,
            gsl::span<const int64_t> strides = {});

  void ReleaseBuffer();

#ifdef ENABLE_STRIDED_TENSORS
  bool CheckIsContiguous() const;
#endif

  std::shared_ptr<Storage> storage_;

  TensorShape shape_;
#ifdef ENABLE_STRIDED_TENSORS
  mutable TensorShapeVector strides_;
  bool is_contiguous_ = true;
#endif

  const PrimitiveDataTypeBase* dtype_;
  OrtMemoryInfo alloc_info_;
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}  // namespace onnxruntime
