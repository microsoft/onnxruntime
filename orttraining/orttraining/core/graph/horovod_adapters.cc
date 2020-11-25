// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/horovod_adapters.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif

namespace onnxruntime {

common::Status ConvertStatus(const hvd::Status& status) {
  switch (status.type()) {
    case hvd::OK:
      return Status::OK();
    case hvd::INVALID_ARGUMENT:
      return common::Status(common::StatusCategory::ONNXRUNTIME, 3 /* MLStatus::InvalidArgument */);
    default:
      return common::Status(common::StatusCategory::ONNXRUNTIME, 1 /* MLStatus::Fail */);
  }
}

hvd::Status ConvertStatus(const common::Status& status) {
  switch (status.Code()) {
    case 0:
      return hvd::Status::OK();
    case 3:
      return hvd::Status::InvalidArgument(status.ErrorMessage());
    default:
      return hvd::Status::UnknownError("Unknown error.");
  }
}

ORTTensor::ORTTensor(const onnxruntime::Tensor* tensor) : tensor_(tensor) {}

const hvd::DataType ORTTensor::dtype() const {
  auto type = tensor_->DataType();
  if (type == DataTypeImpl::GetType<uint8_t>()) {
    return hvd::HOROVOD_UINT8;
  } else if (type == DataTypeImpl::GetType<int8_t>()) {
    return hvd::HOROVOD_INT8;
  } else if (type == DataTypeImpl::GetType<uint16_t>()) {
    return hvd::HOROVOD_UINT16;
  } else if (type == DataTypeImpl::GetType<int16_t>()) {
    return hvd::HOROVOD_INT16;
  } else if (type == DataTypeImpl::GetType<int32_t>()) {
    return hvd::HOROVOD_INT32;
  } else if (type == DataTypeImpl::GetType<int64_t>()) {
    return hvd::HOROVOD_INT64;
  } else if (type == DataTypeImpl::GetType<float>()) {
    return hvd::HOROVOD_FLOAT32;
  } else if (type == DataTypeImpl::GetType<double>()) {
    return hvd::HOROVOD_FLOAT64;
  } else if (type == DataTypeImpl::GetType<bool>()) {
    return hvd::HOROVOD_BOOL;
  } else if (type == DataTypeImpl::GetType<MLFloat16>()) {
    return hvd::HOROVOD_FLOAT16;
  } else {
    throw std::logic_error("Invalid tensor type.");
  }
}

const hvd::ReduceOp GetReduceOp(const int64_t reduce_op_enum) {
  if (reduce_op_enum == hvd::horovod_reduce_op_average()) {
    return hvd::ReduceOp::AVERAGE;
  } else if (reduce_op_enum == hvd::horovod_reduce_op_sum()) {
    return hvd::ReduceOp::SUM;
  } else if (reduce_op_enum == hvd::horovod_reduce_op_adasum()) {
    return hvd::ReduceOp::ADASUM;
  } else {
    throw std::logic_error("Invalid horovod reduce op.");
  }
};

const hvd::TensorShape ORTTensor::shape() const {
  hvd::TensorShape shape;
  const std::vector<int64_t> original_shape = tensor_->Shape().GetDims();
  for (auto dim : original_shape) {
    shape.AddDim(dim);
  }
  return shape;
}

const void* ORTTensor::data() const {
  return tensor_->DataRaw();
}

int64_t ORTTensor::size() const {
  return (int64_t)tensor_->SizeInBytes();
}

ORTPersistentBuffer::ORTPersistentBuffer(AllocatorPtr allocator, int64_t size) : allocator_(allocator) {
  buffer_ = allocator->Alloc(size);
}

ORTPersistentBuffer::~ORTPersistentBuffer() {
  if (buffer_) {
    allocator_->Free(buffer_);
  }
}

const void* ORTPersistentBuffer::AccessData(std::shared_ptr<hvd::OpContext>) const {
  return buffer_;
}

ORTOpContext::ORTOpContext(AllocatorPtr allocator) : allocator_(allocator) {}

hvd::Status ORTOpContext::AllocatePersistent(int64_t size, std::shared_ptr<hvd::PersistentBuffer>* tensor) {
  *tensor = std::make_shared<ORTPersistentBuffer>(allocator_, size);
  return hvd::Status::OK();
}

hvd::Status ORTOpContext::AllocateOutput(hvd::TensorShape /*shape*/, std::shared_ptr<hvd::Tensor>* tensor) {
  *tensor = nullptr;
  return hvd::Status::InvalidArgument("Not implemented");
}

hvd::Status ORTOpContext::AllocateZeros(int64_t /*num_elements*/, hvd::DataType /*dtype*/, std::shared_ptr<hvd::Tensor>* tensor) {
  *tensor = nullptr;
  return hvd::Status::InvalidArgument("Not implemented");
}

hvd::Framework ORTOpContext::framework() const {
  // TODO: create ORT in horovod and change this
  return hvd::Framework::TENSORFLOW;
}

}  // namespace onnxruntime

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
