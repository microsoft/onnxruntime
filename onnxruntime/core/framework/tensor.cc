// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"

#include <utility>
#include "core/framework/allocatormgr.h"
using namespace std;
namespace onnxruntime {

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtAllocatorInfo& alloc,
               int64_t offset)
    : alloc_info_(alloc) {
  ORT_ENFORCE(p_type != nullptr);
  Init(p_type, shape, p_data, nullptr, offset);
}

Tensor::Tensor(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator, int64_t offset)
    : alloc_info_(allocator->Info()) {
  ORT_ENFORCE(p_type != nullptr);
  int64_t shape_size = shape.Size();
  if (shape_size < 0 || static_cast<uint64_t>(shape_size) >= std::numeric_limits<size_t>::max())
    ORT_THROW("shape.Size() must >=0");
  void* p_data = nullptr;
  if(shape_size > 0)
     p_data = allocator->AllocArray(static_cast<size_t>(shape_size), p_type->Size());
  Init(p_type, shape, p_data, allocator, offset);
}

void Tensor::Init(MLDataType p_type, const TensorShape& shape, void* p_raw_data, AllocatorPtr deleter, int64_t offset) {
  int64_t shape_size = shape.Size();
  if (shape_size < 0) ORT_THROW("shape.Size() must >=0");
  dtype_ = p_type;
  shape_ = shape;
  p_data_ = p_raw_data;
  // if caller passed in a deleter, that means this tensor own this buffer
  // we will release the buffer when this tensor is deconstructed.
  buffer_deleter_ = std::move(deleter);
  // for string tensors, if this tensor own the buffer (caller passed in the deleter)
  // do the placement new for strings on pre-allocated buffer.
  if (buffer_deleter_ && dtype_ == DataTypeImpl::GetType<string>()) {
    auto* ptr = static_cast<string*>(p_data_);
    for (int64_t i = 0, n = shape_size; i < n; ++i) {
      new (ptr + i) string();
    }
  }
  byte_offset_ = offset;
}

Tensor::Tensor(Tensor&& other)
    : p_data_(other.p_data_),
      buffer_deleter_(other.buffer_deleter_),
      shape_(other.shape_),
      dtype_(other.dtype_),
      alloc_info_(other.alloc_info_),
      byte_offset_(other.byte_offset_) {
  other.dtype_ = DataTypeImpl::GetType<float>();
  other.shape_ = TensorShape(vector<int64_t>(1, 0));
  other.p_data_ = nullptr;
  other.buffer_deleter_ = nullptr;
  other.byte_offset_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) {
  if (this != &other) {
    ReleaseBuffer();

    dtype_ = other.dtype_;
    shape_ = other.shape_;
    alloc_info_ = other.alloc_info_;
    byte_offset_ = other.byte_offset_;
    p_data_ = other.p_data_;
    buffer_deleter_ = other.buffer_deleter_;

    other.dtype_ = DataTypeImpl::GetType<float>();
    other.shape_ = TensorShape(vector<int64_t>(1, 0));
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
    // if current tensor is responsible for delete the buffer
    // and it is a string tensor, need to explict call string's
    // deconstructor.
    if (dtype_ == DataTypeImpl::GetType<string>()) {
      auto* ptr = static_cast<string*>(p_data_);
      int64_t len = shape_.Size();
      for (int64_t i = 0; i < len; i++)
        ptr[i].~string();
    }
    buffer_deleter_->Free(p_data_);
  }
}

}  // namespace onnxruntime
