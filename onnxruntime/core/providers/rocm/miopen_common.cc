// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "miopen_common.h"
#include "core/common/gsl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"

namespace onnxruntime {
namespace rocm {

namespace {
std::string layoutTypeToString(miopenTensorLayout_t layout) {
  if (layout == MIOPEN_NCHW_LAYOUT) {
    return "NCHW";
  } else if (layout == MIOPEN_NHWC_LAYOUT) {
    return "NHWC";
  } else {
    ORT_THROW("Currently, ORT only supports two MIOpen layout: MIOPEN_NCHW_LAYOUT and MIOPEN_NHWC_LAYOUT.");
  }
}

// This functions was modified from https://github.com/ROCmSoftwarePlatform/MIOpen/src/include/miopen/tensor_layout.hpp
template <typename T>
void tensorLayoutToStrides(const InlinedVector<T>& len,
                           miopenTensorLayout_t len_tensor_layout,
                           miopenTensorLayout_t tensor_layout,
                           InlinedVector<T>& strides) {
  std::string len_layout = layoutTypeToString(len_tensor_layout);
  std::string layout = layoutTypeToString(tensor_layout);
  // Bind the layout and the dimension lengths together into a map.
  std::map<char, T> dim_to_len;
  std::transform(len.begin(),
                 len.end(),
                 len_layout.begin(),
                 std::inserter(dim_to_len, dim_to_len.end()),
                 [](T l, char dim) { return std::make_pair(dim, l); });

  // Now construct the strides according to layout by multiply the
  // dimension lengths together.
  std::transform(len_layout.begin(),
                 len_layout.end(),
                 strides.begin(),
                 [&layout, &dim_to_len](char cur_layout_char) {
                   auto pos = layout.find(cur_layout_char);
                   if (pos == std::string::npos) {
                     ORT_THROW(std::string("mismatched layout string - ").append(layout));
                   }
                   return std::accumulate(layout.begin() + pos + 1,
                                          layout.end(),
                                          1,
                                          [&dim_to_len](T accumulator, char l) {
                                            return accumulator * dim_to_len[l];
                                          });
                 });
}
}  // namespace

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

Status MiopenTensor::Set(gsl::span<const int64_t> input_dims, miopenDataType_t dataType, bool is_nhwc) {
  if (is_nhwc) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "NHWC Tensor usage is not supported in AMD builds for now");
  }

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

Status MiopenTensor::Set(miopenDataType_t dataType, miopenTensorLayout_t tensor_layout, int n, int c, int h, int w) {
  ORT_RETURN_IF_ERROR(CreateTensorIfNeeded());

  // miopenSetNdTensorDescriptorWithLayout doesn't support NHWC layout now.
  // We use miopenSetTensorDescriptor with dims = [N, C, H, W], strides = [N*W*C, 1, W*C, C] for NHWC layout.
  const int num_lens = 4;
  InlinedVector<int> dims = {n, c, h, w};
  InlinedVector<int> strides(num_lens);

  miopenTensorLayout_t len_layout = MIOPEN_NCHW_LAYOUT;
  tensorLayoutToStrides(dims, len_layout, tensor_layout, strides);

  MIOPEN_RETURN_IF_ERROR(miopenSetTensorDescriptor(tensor_, dataType, static_cast<int>(num_lens), dims.data(), strides.data()));
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

Status MiopenTensorDescriptor::Set(miopenDataType_t data_type, miopenTensorLayout_t tensor_layout, int k, int c, int h, int w) {
  if (!desc_)
    MIOPEN_RETURN_IF_ERROR(miopenCreateTensorDescriptor(&desc_));

  // miopenSetNdTensorDescriptorWithLayout doesn't support NHWC layout now.
  // We use miopenSetTensorDescriptor with dims = [N, C, H, W], strides = [N*W*C, 1, W*C, C] for NHWC layout.
  const int num_lens = 4;
  InlinedVector<int> dims = {k, c, h, w};
  InlinedVector<int> strides(num_lens);

  miopenTensorLayout_t len_layout = MIOPEN_NCHW_LAYOUT;
  tensorLayoutToStrides(dims, len_layout, tensor_layout, strides);

  MIOPEN_RETURN_IF_ERROR(miopenSetTensorDescriptor(desc_, data_type, static_cast<int>(num_lens), dims.data(), strides.data()));
  return Status::OK();
}

template <typename ElemType>
miopenDataType_t MiopenTensor::GetDataType() {
  ORT_THROW("miopen engine currently supports only single/half/int32/int8 precision data types.");
}

#if ROCM_VERSION >= 50000
template <>
miopenDataType_t MiopenTensor::GetDataType<double>() {
  return miopenDouble;
}
#endif

template <>
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
