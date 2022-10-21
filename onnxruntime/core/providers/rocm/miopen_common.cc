// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "miopen_common.h"
#include "core/common/gsl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"

namespace onnxruntime {
namespace rocm {

MiopenTensor::MiopenTensor()
    : tensor_(nullptr) {
}

MiopenTensor::~MiopenTensor() {
  if (tensor_ != nullptr) {
    miopenDestroyTensorDescriptor(tensor_);
    tensor_ = nullptr;
  }
}

Status MiopenTensor::CreateTensorIfNeeded() {
  if (!tensor_)
    MIOPEN_RETURN_IF_ERROR(miopenCreateTensorDescriptor(&tensor_));
  return Status::OK();
}

Status MiopenTensor::Set(gsl::span<const int64_t> input_dims, miopenDataType_t dataType) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());

  int rank = gsl::narrow_cast<int>(input_dims.size());
  TensorPitches pitches(input_dims);
  InlinedVector<int> dims(rank);
  InlinedVector<int> strides(rank);
  for (int i = 0; i < rank; i++) {
    dims[i] = gsl::narrow_cast<int>(input_dims[i]);
    strides[i] = gsl::narrow_cast<int>(pitches[i]);
  }
  MIOPEN_RETURN_IF_ERROR(miopenSetTensorDescriptor(tensor_, dataType, static_cast<int>(rank), dims.data(), strides.data()));
  return Status::OK();
}

Status MiopenTensor::Set(const MiopenTensor& x_desc, miopenBatchNormMode_t mode) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());
  MIOPEN_RETURN_IF_ERROR(miopenDeriveBNTensorDescriptor(tensor_, x_desc, mode));
  return Status::OK();
}

MiopenTensorDescriptor::MiopenTensorDescriptor() : desc_(nullptr) {
  miopenCreateTensorDescriptor(&desc_);
}

MiopenTensorDescriptor::~MiopenTensorDescriptor() {
  if (desc_ != nullptr) {
    miopenCreateTensorDescriptor(&desc_);
    desc_ = nullptr;
  }
}

Status MiopenTensorDescriptor::Set(gsl::span<const int64_t> filter_dims, miopenDataType_t data_type) {
  if (!desc_)
    MIOPEN_RETURN_IF_ERROR(miopenCreateTensorDescriptor(&desc_));

  int rank = gsl::narrow_cast<int>(filter_dims.size());
  InlinedVector<int> w_dims(rank);
  for (int i = 0; i < rank; i++) {
    w_dims[i] = gsl::narrow_cast<int>(filter_dims[i]);
  }

  MIOPEN_RETURN_IF_ERROR(miopenSetTensorDescriptor(desc_,
                                                   data_type,
                                                   rank,
                                                   w_dims.data(),
						   nullptr));
  return Status::OK();
}

template <typename ElemType>
miopenDataType_t MiopenTensor::GetDataType() {
  ORT_THROW("miopen engine currently supports only single/half/int32/int8 precision data types.");
}

#if ROCM_VERSION >= 50000
template<>
miopenDataType_t MiopenTensor::GetDataType<double>() {
  return miopenDouble;
}
#endif

template<>
miopenDataType_t MiopenTensor::GetDataType<float>() {
  return miopenFloat;
}

template <>
miopenDataType_t MiopenTensor::GetDataType<half>() {
  return miopenHalf;
}

template <>
miopenDataType_t MiopenTensor::GetDataType<BFloat16>() {
  ORT_THROW("miopen doesn't support BFloat16.");
  return miopenFloat;
}

template <>
miopenDataType_t MiopenTensor::GetDataType<int32_t>() {
  return miopenInt32;
}

template <>
miopenDataType_t MiopenTensor::GetDataType<int8_t>() {
  return miopenInt8;
}

}  // namespace rocm
}  // namespace onnxruntime
