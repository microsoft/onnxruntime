// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"

#include <utility>
#include "core/common/safeint.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/framework/utils.h"

namespace onnxruntime {

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc,
               ptrdiff_t offset)
    : alloc_info_(alloc) {
  ORT_ENFORCE(p_type != nullptr);
  Init(p_type, shape, p_data, nullptr, offset);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator)
    : alloc_info_(allocator->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  int64_t shape_size = shape.Size();  // value returned is checked for overflow by TensorShape::Size()
  if (shape_size < 0)
    ORT_THROW("shape.Size() must >=0");

  void* p_data = nullptr;
  if (shape_size > 0) {
    SafeInt<size_t> len = 0;
    if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(shape_size), p_type->Size(), &len))
      ORT_THROW("tensor failed memory size calculation");

    p_data = allocator->Alloc(len);
  }

  Init(p_type, shape, p_data, allocator);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, std::shared_ptr<IAllocator> deleter,
               ptrdiff_t offset)
    : alloc_info_(deleter->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  Init(p_type, shape, p_data, deleter, offset);
}

void Tensor::InitOrtValue(MLDataType elt_type, const TensorShape& shape,
                          std::shared_ptr<IAllocator> allocator, OrtValue& ort_value) {
  auto p_tensor = std::make_unique<Tensor>(elt_type, shape, std::move(allocator));
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

void Tensor::InitOrtValue(MLDataType p_type, const TensorShape& shape, void* p_data,
                          const OrtMemoryInfo& location, OrtValue& ort_value) {
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  auto p_tensor = std::make_unique<Tensor>(p_type, shape, p_data, location);
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
}

size_t Tensor::SizeInBytes() const {
  size_t ret;
  if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(shape_.Size()), dtype_->Size(), &ret)) {
    ORT_THROW("tensor size overflow");
  }
  return ret;
}

void Tensor::Init(MLDataType p_type, const TensorShape& shape, void* p_raw_data, AllocatorPtr deleter, ptrdiff_t offset) {
  int64_t shape_size = shape.Size();
  if (shape_size < 0) ORT_THROW("shape.Size() must >=0");
  dtype_ = p_type->AsPrimitiveDataType();
  ORT_ENFORCE(dtype_ != nullptr, "Tensor is expected to contain one of the primitive data types. Got: ",
              DataTypeImpl::ToString(p_type));
  shape_ = shape;
  p_data_ = p_raw_data;
  // if caller passed in a deleter, that means this tensor own this buffer
  // we will release the buffer when this tensor is deconstructed.
  buffer_deleter_ = std::move(deleter);
  // for string tensors, if this tensor own the buffer (caller passed in the deleter)
  // do the placement new for strings on pre-allocated buffer.
  if (buffer_deleter_ && IsDataTypeString()) {
    utils::ConstructStrings(p_data_, shape_size);
  }
  byte_offset_ = offset;
}

Tensor::Tensor(Tensor&& other) noexcept
    : p_data_(other.p_data_),
      buffer_deleter_(other.buffer_deleter_),
      shape_(other.shape_),
      dtype_(other.dtype_),
      alloc_info_(other.alloc_info_),
      byte_offset_(other.byte_offset_) {
  other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
  other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
  other.p_data_ = nullptr;
  other.buffer_deleter_ = nullptr;
  other.byte_offset_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    ReleaseBuffer();

    dtype_ = other.dtype_;
    shape_ = other.shape_;
    alloc_info_ = other.alloc_info_;
    byte_offset_ = other.byte_offset_;
    p_data_ = other.p_data_;
    buffer_deleter_ = other.buffer_deleter_;

    other.dtype_ = DataTypeImpl::GetType<float>()->AsPrimitiveDataType();
    other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
    other.p_data_ = nullptr;
    other.byte_offset_ = 0;
    other.buffer_deleter_ = nullptr;
  }
  return *this;
}

Tensor::~Tensor() {
  ReleaseBuffer();
}

void Tensor::ReleaseBuffer() {
  if (buffer_deleter_) {
    if (IsDataTypeString()) {
      utils::DestroyStrings(p_data_, shape_.Size());
    }
    buffer_deleter_->Free(p_data_);
  }
}

}  // namespace onnxruntime
