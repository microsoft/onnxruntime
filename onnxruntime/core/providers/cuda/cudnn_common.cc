// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include <utility>

#include "core/providers/cuda/cudnn_common.h"
#include "core/common/inlined_containers.h"
#include "core/common/gsl.h"
#include "shared_inc/cuda_call.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

CudnnTensor::CudnnTensor()
    : tensor_(nullptr) {
}

CudnnTensor::~CudnnTensor() {
  if (tensor_ != nullptr) {
    cudnnDestroyTensorDescriptor(tensor_);
    tensor_ = nullptr;
  }
}

Status CudnnTensor::CreateTensorIfNeeded() {
  if (!tensor_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateTensorDescriptor(&tensor_));
  return Status::OK();
}

Status CudnnTensor::Set(gsl::span<const int64_t> input_dims, cudnnDataType_t dataType, bool is_nhwc) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());

  int rank = gsl::narrow_cast<int>(input_dims.size());
  TensorPitches pitches(input_dims);
  InlinedVector<int, kTensorShapeSmallBufferElementsSize> dims(rank);
  InlinedVector<int, kTensorShapeSmallBufferElementsSize> strides(rank);
  for (int i = 0; i < rank; i++) {
    dims[i] = gsl::narrow_cast<int>(input_dims[i]);
    strides[i] = gsl::narrow_cast<int>(pitches[i]);
  }
  if (is_nhwc) {
    std::swap(dims[1], dims[rank - 1]);
    std::swap(strides[1], strides[rank - 1]);
  }
  CUDNN_RETURN_IF_ERROR(cudnnSetTensorNdDescriptor(tensor_, dataType, static_cast<int>(rank), dims.data(), strides.data()));
  return Status::OK();
}

Status CudnnTensor::Set(cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());
  CUDNN_RETURN_IF_ERROR(cudnnSetTensor4dDescriptor(tensor_, format, dataType, n, c, h, w));
  return Status::OK();
}

Status CudnnTensor::Set(const CudnnTensor& x_desc, cudnnBatchNormMode_t mode) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());
  CUDNN_RETURN_IF_ERROR(cudnnDeriveBNTensorDescriptor(tensor_, x_desc, mode));
  return Status::OK();
}

CudnnDataTensor::CudnnDataTensor()
    : tensor_(nullptr) {
}

CudnnDataTensor::~CudnnDataTensor() {
  if (tensor_ != nullptr) {
    cudnnDestroyRNNDataDescriptor(tensor_);
    tensor_ = nullptr;
  }
}

Status CudnnDataTensor::CreateTensorIfNeeded() {
  if (!tensor_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateRNNDataDescriptor(&tensor_));
  return Status::OK();
}

Status CudnnDataTensor::Set(cudnnDataType_t dataType,
                            int64_t max_seq_length,
                            int64_t batch_size,
                            int64_t data_size,
                            const int32_t* seq_lengths) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());

  // CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED works with CUDNN_RNN_PADDED_IO_ENABLED, so that it will auto fill 0 for the shorter sequences
  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  float padding_fill = 0.0f;
  CUDNN_RETURN_IF_ERROR(cudnnSetRNNDataDescriptor(tensor_, dataType, layout,
                                                  static_cast<int>(max_seq_length),
                                                  static_cast<int>(batch_size),
                                                  static_cast<int>(data_size),
                                                  seq_lengths,
                                                  static_cast<void*>(&padding_fill)));
  return Status::OK();
}

CudnnFilterDescriptor::CudnnFilterDescriptor() : desc_(nullptr) {
  cudnnCreateFilterDescriptor(&desc_);
}

CudnnFilterDescriptor::~CudnnFilterDescriptor() {
  if (desc_ != nullptr) {
    cudnnDestroyFilterDescriptor(desc_);
    desc_ = nullptr;
  }
}

Status CudnnFilterDescriptor::Set(gsl::span<const int64_t> filter_dims, cudnnDataType_t data_type) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateFilterDescriptor(&desc_));

  int rank = gsl::narrow_cast<int>(filter_dims.size());
  InlinedVector<int> w_dims(rank);
  for (int i = 0; i < rank; i++) {
    w_dims[i] = gsl::narrow_cast<int>(filter_dims[i]);
  }

  CUDNN_RETURN_IF_ERROR(cudnnSetFilterNdDescriptor(desc_,
                                                   data_type,
                                                   CUDNN_TENSOR_NCHW,
                                                   rank,
                                                   w_dims.data()));
  return Status::OK();
}

Status CudnnFilterDescriptor::Set(cudnnTensorFormat_t format, cudnnDataType_t dataType, int k, int c, int h, int w) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateFilterDescriptor(&desc_));

  CUDNN_RETURN_IF_ERROR(cudnnSetFilter4dDescriptor(desc_, dataType, format, k, c, h, w));
  return Status::OK();
}

template <typename ElemType>
cudnnDataType_t CudnnTensor::GetDataType() {
  ORT_THROW("cuDNN engine currently supports only single/double/half/int8/uint8 precision data types. Got:",
            typeid(ElemType).name());
  // Not reachable but GCC complains
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<float>() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<double>() {
  return CUDNN_DATA_DOUBLE;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<half>() {
  return CUDNN_DATA_HALF;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<BFloat16>() {
  ORT_THROW("cuDNN doesn't support BFloat16.");
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<int8_t>() {
  return CUDNN_DATA_INT8;
}

template <>
cudnnDataType_t CudnnTensor::GetDataType<uint8_t>() {
  return CUDNN_DATA_UINT8;
}

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

const float Consts<BFloat16>::Zero = 0;
const float Consts<BFloat16>::One = 1;

template <>
const int8_t Consts<int8_t>::Zero = 0;

template <>
const int8_t Consts<int8_t>::One = 1;

template <>
const uint8_t Consts<uint8_t>::Zero = 0;

template <>
const uint8_t Consts<uint8_t>::One = 1;

#if !defined(DISABLE_FLOAT8_TYPES)

template <>
const Float8E4M3FN Consts<Float8E4M3FN>::Zero = Float8E4M3FN(0.0f, true);

template <>
const Float8E4M3FN Consts<Float8E4M3FN>::One = Float8E4M3FN(1.0f, true);

template <>
const Float8E5M2 Consts<Float8E5M2>::Zero = Float8E5M2(0.0f, true);

template <>
const Float8E5M2 Consts<Float8E5M2>::One = Float8E5M2(1.0f, true);

#endif

}  // namespace cuda
}  // namespace onnxruntime
