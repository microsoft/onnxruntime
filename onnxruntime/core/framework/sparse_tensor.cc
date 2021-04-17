// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/data_transfer_manager.h"

using namespace onnxruntime::common;

namespace onnxruntime {

std::ostream& operator<<(std::ostream& os, SparseFormatFlags flags) {
  return os << std::hex << static_cast<uint32_t>(flags);
}

// Group member initialization in one place
// private helper __ctor
SparseTensor::SparseTensor(MLDataType elt_type,
                           const TensorShape& dense_shape,
                           const OrtMemoryInfo& location,
                           std::shared_ptr<IAllocator>&& allocator)
    : format_flags_(SparseFormatFlags::kUndefined),
      dense_shape_(dense_shape),
      allocator_(std::move(allocator)),
      location_(location),
      ml_data_type_(elt_type->AsPrimitiveDataType()),
      rep_() {
  ORT_ENFORCE(ml_data_type_ != nullptr, "Expecting a PrimitiveDataType as elt_type");
}

SparseTensor& SparseTensor::operator=(SparseTensor&& o) noexcept {
  format_flags_ = o.format_flags_;
  dense_shape_ = std::move(o.dense_shape_);
  allocator_ = std::move(o.allocator_);
  location_ = std::move(o.location_);
  ml_data_type_ = std::move(o.ml_data_type_);
  rep_ = std::move(o.rep_);
  return *this;
}

SparseTensor::~SparseTensor() = default;

SparseRep::SparseRep(MLDataType data_type, const TensorShape& values_shape,
                     int64_t total_buffer_size, const AllocatorPtr& allocator)
    : allocator_(allocator), 
      p_data_(nullptr),
      values_([&]() mutable -> Tensor {
        // In the edge case when tensor is absolutely sparse, we do not have neither values, nor indices
        if (total_buffer_size > 0) {
          ORT_ENFORCE(values_shape.Size() * static_cast<int64_t>(data_type->Size()) < total_buffer_size,
                      "Values size must be less than total buffer size");
          auto data_ptr = IAllocator::MakeUniquePtr<void>(allocator_, total_buffer_size);
          ORT_ENFORCE(data_ptr != nullptr, "SparseTensor Allocation failed for size: ", total_buffer_size);
          // We own the buffer, so we must properly construct strings. Neither of the Tensors
          // we construct on top of the buffer own it. We are constructing empty strings, hopefully
          // nothrow and no buffer allocation
          if (utils::IsDataTypeString(data_type)) {
            const int64_t shape_size = values_shape.Size();
            auto* ptr = static_cast<std::string*>(data_ptr.get());
            for (int64_t i = 0, n = shape_size; i < n; ++i) {
              new (ptr + i) std::string();
            }
          }
          p_data_ = data_ptr.release();
        }
        return Tensor(data_type, values_shape, p_data_, allocator_->Info());
      }()) {
}

SparseRep::SparseRep(MLDataType data_type, const TensorShape& values_shape,
                     void* values, const OrtMemoryInfo& location)
    : allocator_(),
      p_data_(nullptr),
      values_(data_type, values_shape, values, location) {
}

void SparseRep::ReleaseBuffer() {
  if (allocator_ && p_data_ != nullptr) {
    // if current tensor is responsible for deleting the buffer
    // and it is a string tensor, need to explicitly call string(s)
    // __dtor(s).
    if (values_.IsDataTypeString()) {
      using string = std::string;
      auto* ptr = static_cast<std::string*>(p_data_);
      int64_t len = values_.Shape().Size();
      for (int64_t i = 0; i < len; i++)
        ptr[i].~string();
    }
    allocator_->Free(p_data_);
  }
}

SparseRep::~SparseRep() {
  ReleaseBuffer();
}

Status SparseTensor::Copy(const DataTransferManager& data_transfer_manager, int exec_q_id, SparseTensor& dst_tensor) const {
  // Do not copy same destination
  if (this == &dst_tensor) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(format_flags_ != SparseFormatFlags::kUndefined, "This instance should not be empty");
  ORT_RETURN_IF_NOT(rep_ != nullptr, "This instance should not be empty");
  ORT_RETURN_IF_NOT(dst_tensor.FormatFlags() == SparseFormatFlags::kUndefined, "Destination should be empty");
  ORT_RETURN_IF_NOT(dst_tensor.allocator_ != nullptr, "Destination must have an allocator set");
  ORT_RETURN_IF_NOT(dst_tensor.dense_shape_.Size() >= dense_shape_.Size(), "Destination must have enough space");

  std::unique_ptr<SparseRep> rep_copy;
  ORT_RETURN_IF_ERROR(rep_->Copy(data_transfer_manager, dst_tensor.allocator_, exec_q_id, rep_copy));

  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(Values(), rep_copy->MutableValues(), exec_q_id));
  dst_tensor.rep_ = std::move(rep_copy);
  dst_tensor.format_flags_ = format_flags_;
  // Allocator, location, data type and shape must have been already set
  return Status::OK();
}

Status SparseTensor::Copy(const IDataTransfer& data_transfer, SparseTensor& dst_tensor, int exec_q_id) const {
  // Do not copy same destination
  if (this == &dst_tensor) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(format_flags_ != SparseFormatFlags::kUndefined, "This instance should not be empty");
  ORT_RETURN_IF_NOT(rep_ != nullptr, "This instance should not be empty");
  ORT_RETURN_IF_NOT(dst_tensor.FormatFlags() == SparseFormatFlags::kUndefined, "Destination should be empty");
  ORT_RETURN_IF_NOT(dst_tensor.allocator_ != nullptr, "Destination must have a CPU allocator set");
  ORT_RETURN_IF_NOT(dst_tensor.dense_shape_.Size() == dense_shape_.Size(), "Must have the same shape");
  std::unique_ptr<SparseRep> rep_copy;
  ORT_RETURN_IF_ERROR(rep_->Copy(data_transfer, dst_tensor.allocator_, exec_q_id, rep_copy));
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(Values(), rep_copy->MutableValues(), exec_q_id));
  dst_tensor.rep_ = std::move(rep_copy);
  dst_tensor.format_flags_ = format_flags_;
  // Allocator and shape must have been already set
  return Status::OK();
}

}  // namespace onnxruntime
