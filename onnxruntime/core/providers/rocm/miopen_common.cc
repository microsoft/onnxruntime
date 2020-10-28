// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "miopen_common.h"
#include "gsl/gsl"
#include "shared_inc/rocm_call.h"
#include "core/providers/cpu/tensor/utils.h"

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

Status MiopenTensor::Set(const std::vector<int64_t>& input_dims, miopenDataType_t dataType) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());

  int rank = gsl::narrow_cast<int>(input_dims.size());
  TensorPitches pitches(input_dims);
  std::vector<int> dims(rank);
  std::vector<int> strides(rank);
  for (int i = 0; i < rank; i++) {
    dims[i] = gsl::narrow_cast<int>(input_dims[i]);
    strides[i] = gsl::narrow_cast<int>(pitches[i]);
  }
  MIOPEN_RETURN_IF_ERROR(miopenSetTensorDescriptor(tensor_, dataType, static_cast<int>(rank), dims.data(), strides.data()));
  return Status::OK();
}

template <typename ElemType>
miopenDataType_t MiopenTensor::GetDataType() {
  ORT_THROW("miopen engine currently supports only single/half precision data types.");
}

template <>
miopenDataType_t MiopenTensor::GetDataType<float>() { return miopenFloat; }

template <>
miopenDataType_t MiopenTensor::GetDataType<half>() { return miopenHalf; }

template <>
const float Consts<float>::One = 1;

template <>
const double Consts<double>::One = 1;

template <>
const float Consts<float>::Zero = 0;

template <>
const double Consts<double>::Zero = 0;

const float Consts<half>::Zero = 0;

const float Consts<half>::One = 1;

}  // namespace rocm
}  // namespace onnxruntime
