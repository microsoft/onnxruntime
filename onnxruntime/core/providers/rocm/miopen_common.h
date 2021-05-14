// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <miopen/miopen.h>

#include "rocm_common.h"
#include "core/framework/tensor.h"
#include <cfloat>

namespace onnxruntime {
namespace rocm {

class MiopenTensor final {
 public:
  MiopenTensor();
  ~MiopenTensor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MiopenTensor);

  Status Set(const std::vector<int64_t>& input_dims, miopenDataType_t dataType);
  Status Set(const MiopenTensor& x_desc, miopenBatchNormMode_t mode);

  operator miopenTensorDescriptor_t() const { return tensor_; }

  template <typename T>
  static miopenDataType_t GetDataType();

 private:
  Status CreateTensorIfNeeded();

  miopenTensorDescriptor_t tensor_;
};

template <typename ElemType>
struct Consts {
  static const ElemType Zero;
  static const ElemType One;
};

template <>
struct Consts<half> {
  static const float Zero;
  static const float One;
};

}  // namespace rocm
}  // namespace onnxruntime
