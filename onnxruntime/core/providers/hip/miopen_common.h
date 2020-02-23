// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <miopen/miopen.h>

#include "hip_common.h"
#include "core/framework/tensor.h"
#include <cfloat>

namespace onnxruntime {
namespace hip {

class MiopenTensor final {
 public:
  MiopenTensor();
  ~MiopenTensor();

  Status Set(const std::vector<int64_t>& input_dims, miopenDataType_t dataType);
  Status Set(const MiopenTensor& x_desc, miopenBatchNormMode_t mode);

  operator miopenTensorDescriptor_t() const { return tensor_; }

  template <typename T>
  static miopenDataType_t GetDataType();

 private:
  Status CreateTensorIfNeeded();

  miopenTensorDescriptor_t tensor_;
};

// class MiopenDataTensor final {
//  public:
//   MiopenDataTensor();
//   ~MiopenDataTensor();

//   Status Set(miopenDataType_t dataType,
//              int64_t max_seq_length,
//              int64_t batch_size,
//              int64_t data_size,
//              const int32_t* seq_lengths);

//   operator miopenRNNDataDescriptor_t() const { return tensor_; }

//  private:
//   Status CreateTensorIfNeeded();

//   miopenRNNDataDescriptor_t tensor_;
// };

// class MiopenFilterDescriptor final {
//  public:
//   MiopenFilterDescriptor();
//   ~MiopenFilterDescriptor();

//   Status Set(const std::vector<int64_t>& filter_dims, miopenDataType_t data_typ);

//   operator miopenFilterDescriptor_t() const { return desc_; }

//  private:
//   miopenFilterDescriptor_t desc_;
// };

// class MiopenDropout final {
//  public:
//   MiopenDropout() : dropout_desc_(nullptr) {
//   }

//   Status GetMiopenDropoutStatesSize(const miopenHandle_t& miopenHandle, size_t& stateSize) {
//     MIOPEN_RETURN_IF_ERROR(miopenDropoutGetStatesSize(miopenHandle, &stateSize));

//     return Status::OK();
//   }

//   Status Set(const miopenHandle_t& miopenHandle,
//              void* states,
//              size_t stateSize,
//              float dropout = 0.0f,
//              unsigned long long seed = 1) {
//     ORT_RETURN_IF_ERROR(CreateDescriptorIfNeeded());
//     MIOPEN_RETURN_IF_ERROR(miopenSetDropoutDescriptor(dropout_desc_,
//                                                     miopenHandle,
//                                                     dropout,
//                                                     states,
//                                                     stateSize,
//                                                     seed));

//     return Status::OK();
//   }

//   ~MiopenDropout() {
//     if (dropout_desc_ != nullptr) {
//       miopenDestroyDropoutDescriptor(dropout_desc_);
//     }
//   }

//   operator miopenDropoutDescriptor_t() const {
//     return dropout_desc_;
//   }

//   Status CreateDescriptorIfNeeded() {
//     if (!dropout_desc_)
//       MIOPEN_RETURN_IF_ERROR(miopenCreateDropoutDescriptor(&dropout_desc_));
//     return Status::OK();
//   }

//  private:
//   miopenDropoutDescriptor_t dropout_desc_;
// };

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

// inline double ClampMiopenBatchNormEpsilon(double epsilon) {
//   if (epsilon < MIOPEN_BN_MIN_EPSILON) {
//     if (MIOPEN_BN_MIN_EPSILON - epsilon > FLT_EPSILON)
//       LOGS_DEFAULT(WARNING) << "Provided epsilon is smaller than MIOPEN_BN_MIN_EPSILON. Setting it to MIOPEN_BN_MIN_EPSILON";
//     return MIOPEN_BN_MIN_EPSILON;
//   }
//   return epsilon;
// }

}  // namespace hip
}  // namespace onnxruntime
