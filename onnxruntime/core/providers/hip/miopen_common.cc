// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "miopen_common.h"
#include "gsl/gsl"
#include "hip_call.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace hip {

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

// Status MiopenTensor::Set(const MiopenTensor& x_desc, miopenBatchNormMode_t mode) {
//   ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());
//   MIOPEN_RETURN_IF_ERROR(miopenDeriveBNTensorDescriptor(tensor_, x_desc, mode));
//   return Status::OK();
// }

template <typename ElemType>
miopenDataType_t MiopenTensor::GetDataType() {
  if (typeid(ElemType) == typeid(float))
    return miopenFloat;
  else if (typeid(ElemType) == typeid(half))
    return miopenHalf;
  else
    ORT_THROW("miopen engine currently supports only single/half precision data types.");
}

// MiopenDataTensor::MiopenDataTensor()
//     : tensor_(nullptr) {
// }

// MiopenDataTensor::~MiopenDataTensor() {
//   if (tensor_ != nullptr) {
//     miopenDestroyRNNDataDescriptor(tensor_);
//     tensor_ = nullptr;
//   }
// }

// Status MiopenDataTensor::CreateTensorIfNeeded() {
//   if (!tensor_)
//     MIOPEN_RETURN_IF_ERROR(miopenCreateRNNDataDescriptor(&tensor_));
//   return Status::OK();
// }

// Status MiopenDataTensor::Set(miopenDataType_t dataType,
//                             int64_t max_seq_length,
//                             int64_t batch_size,
//                             int64_t data_size,
//                             const int32_t* seq_lengths) {
//   ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());

//   // MIOPEN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED works with MIOPEN_RNN_PADDED_IO_ENABLED, so that it will auto fill 0 for the shorter sequences
//   miopenRNNDataLayout_t layout = MIOPEN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
//   float padding_fill = 0.0f;
//   MIOPEN_RETURN_IF_ERROR(miopenSetRNNDataDescriptor(tensor_, dataType, layout,
//                                                   static_cast<int>(max_seq_length),
//                                                   static_cast<int>(batch_size),
//                                                   static_cast<int>(data_size),
//                                                   seq_lengths,
//                                                   static_cast<void*>(&padding_fill)));
//   return Status::OK();
// }

// MiopenFilterDescriptor::MiopenFilterDescriptor() : desc_(nullptr) {
//   miopenCreateFilterDescriptor(&desc_);
// }

// MiopenFilterDescriptor::~MiopenFilterDescriptor() {
//   if (desc_ != nullptr) {
//     miopenDestroyFilterDescriptor(desc_);
//     desc_ = nullptr;
//   }
// }

// Status MiopenFilterDescriptor::Set(const std::vector<int64_t>& filter_dims, miopenDataType_t data_type) {
//   if (!desc_)
//     MIOPEN_RETURN_IF_ERROR(miopenCreateFilterDescriptor(&desc_));

//   int rank = gsl::narrow_cast<int>(filter_dims.size());
//   std::vector<int> w_dims(rank);
//   for (int i = 0; i < rank; i++) {
//     w_dims[i] = gsl::narrow_cast<int>(filter_dims[i]);
//   }

//   MIOPEN_RETURN_IF_ERROR(miopenSetFilterNdDescriptor(desc_,
//                                                    data_type,
//                                                    MIOPEN_TENSOR_NCHW,
//                                                    rank,
//                                                    w_dims.data()));
//   return Status::OK();
// }

template miopenDataType_t MiopenTensor::GetDataType<float>();
template miopenDataType_t MiopenTensor::GetDataType<double>();
template miopenDataType_t MiopenTensor::GetDataType<half>();

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

}  // namespace hip
}  // namespace onnxruntime
