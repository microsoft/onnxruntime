// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <miopen/miopen.h>

#include "rocm_common.h"
#include "core/framework/tensor.h"
#include <cfloat>

const double MIOPEN_BN_MIN_EPSILON = 1e-5;

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

template <typename ElemType>
struct ReduceConsts {
  static const ElemType Zero;
  static const ElemType One;
};

#if ROCM_VERSION >= 40300
// Up until ROCm 4.2 miopenReduceTensor() required alpha/beta to be the same data
// type as the input type. This differs from cudnnReduceTensor() and other
// MIOpen/cuDNN APIs where alpha/beta are float when input type is half (float16).
template <>
struct ReduceConsts<half> {
  static const float Zero;
  static const float One;
};
#endif

inline double ClampMiopenBatchNormEpsilon(double epsilon) {
  if (epsilon < MIOPEN_BN_MIN_EPSILON) {
    if (MIOPEN_BN_MIN_EPSILON - epsilon > FLT_EPSILON)
      LOGS_DEFAULT(WARNING) << "Provided epsilon is smaller than CUDNN_BN_MIN_EPSILON. Setting it to CUDNN_BN_MIN_EPSILON";
    return MIOPEN_BN_MIN_EPSILON;
  }
  return epsilon;
}

}  // namespace rocm
}  // namespace onnxruntime
