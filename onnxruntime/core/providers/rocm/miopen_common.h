// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cfloat>

#include "core/providers/rocm/rocm_common.h"

#include <miopen/miopen.h>

const double MIOPEN_BN_MIN_EPSILON = 1e-5;

namespace onnxruntime {
namespace rocm {

#define MIOPEN_CONVOLUTION_FWD_ALGO_COUNT 6
#define MIOPEN_CONVOLUTION_BWD_FILTER_ALGO_COUNT 4
#define MIOPEN_CONVOLUTION_BWD_DATA_ALGO_COUNT 6

class MiopenTensor final {
 public:
  MiopenTensor();
  ~MiopenTensor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MiopenTensor);

  Status Set(gsl::span<const int64_t> input_dims, miopenDataType_t dataType);
  Status Set(const MiopenTensor& x_desc, miopenBatchNormMode_t mode);

  operator miopenTensorDescriptor_t() const { return tensor_; }

  template <typename T>
  static miopenDataType_t GetDataType();

 private:
  Status CreateTensorIfNeeded();

  miopenTensorDescriptor_t tensor_;
};

class MiopenTensorDescriptor final {
 public:
  MiopenTensorDescriptor();
  ~MiopenTensorDescriptor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MiopenTensorDescriptor);

  Status Set(gsl::span<const int64_t> filter_dims, miopenDataType_t data_typ);

  operator miopenTensorDescriptor_t() const { return desc_; }

 private:
  miopenTensorDescriptor_t desc_;
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

template <>
struct Consts<BFloat16> {
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

template <>
struct ReduceConsts<BFloat16> {
  static const float Zero;
  static const float One;
};
#endif

inline double ClampMiopenBatchNormEpsilon(double epsilon) {
  if (epsilon < MIOPEN_BN_MIN_EPSILON) {
    if (MIOPEN_BN_MIN_EPSILON - epsilon > FLT_EPSILON)
      LOGS_DEFAULT(WARNING) << "Provided epsilon is smaller than MIOPEN_BN_MIN_EPSILON. Setting it to MIOPEN_BN_MIN_EPSILON";
    return MIOPEN_BN_MIN_EPSILON;
  }
  return epsilon;
}

// Compatitable layer to address the non const arguments in miopenBatchNormalizationForwardTraining
inline miopenStatus_t
miCompatBatchNormalizationForwardTraining(miopenHandle_t handle,
                                          miopenBatchNormMode_t bn_mode,
                                          const void* alpha,
                                          const void* beta,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenTensorDescriptor_t yDesc,
                                          void* y,
                                          const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                          const void* bnScale,
                                          const void* bnBias,
                                          double expAvgFactor,
                                          void* resultRunningMean,
                                          void* resultRunningVariance,
                                          double epsilon,
                                          void* resultSaveMean,
                                          void* resultSaveInvVariance)
{
    return miopenBatchNormalizationForwardTraining(handle,
                                                   bn_mode,
                                                   const_cast<void*>(alpha),
                                                   const_cast<void*>(beta),
                                                   xDesc,
                                                   x,
                                                   yDesc,
                                                   y,
                                                   bnScaleBiasMeanVarDesc,
                                                   const_cast<void*>(bnScale),
                                                   const_cast<void*>(bnBias),
                                                   expAvgFactor,
                                                   resultRunningMean,
                                                   resultRunningVariance,
                                                   epsilon,
                                                   resultSaveMean,
                                                   resultSaveInvVariance);
}


template <typename ScalingFactorType>
miopenStatus_t
miCompatBatchNormalizationForwardInference(miopenHandle_t handle,
                                           miopenBatchNormMode_t bn_mode,
                                           const ScalingFactorType* alpha,
                                           const ScalingFactorType* beta,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                           const void* bnScale,
                                           const void* bnBias,
                                           const void* estimatedMean,
                                           const void* estimatedVariance,
                                           double epsilon)
{
    float compat_alpha = *alpha;
    float compat_beta = *beta;
    // Current MIOpen assumes alpha and beta are fp32
    return miopenBatchNormalizationForwardInference(handle,
                                                    bn_mode,
                                                    &compat_alpha,
                                                    &compat_beta,
                                                    xDesc,
                                                    const_cast<void*>(x),
                                                    yDesc,
                                                    y,
                                                    bnScaleBiasMeanVarDesc,
                                                    const_cast<void*>(bnScale),
                                                    const_cast<void*>(bnBias),
                                                    const_cast<void*>(estimatedMean),
                                                    const_cast<void*>(estimatedVariance),
                                                    epsilon);
}

inline miopenStatus_t
miCompatLRNCrossChannelForward(miopenHandle_t handle,
                               miopenLRNDescriptor_t lrnDesc,
                               miopenLRNMode_t lrnMode,
                               const void* alpha,
                               const miopenTensorDescriptor_t xDesc,
                               const void* x,
                               const void* beta,
                               const miopenTensorDescriptor_t yDesc,
                               void *y)
{
    if (lrnMode != miopenLRNCrossChannel) {
        LOGS_DEFAULT(ERROR) << __func__ << " must be called with lrnMode == miopenLRNCrossChannel";
        return miopenStatusBadParm;
    }
    return miopenLRNForward(handle,
                            lrnDesc,
                            alpha,
                            xDesc,
                            x,
                            beta,
                            yDesc,
                            y,
                            false, // Has not found cudnnLRNCrossChannelBackward anywhere yet
                            nullptr);
}

inline miopenStatus_t
miCompatSetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc,
                         unsigned int lrnN,
                         double lrnAlpha,
                         double lrnBeta,
                         double lrnK)
{
    return miopenSetLRNDescriptor(lrnDesc,
                                  miopenLRNCrossChannel,
                                  lrnN,
                                  lrnAlpha,
                                  lrnBeta,
                                  lrnK);
}

}  // namespace rocm
}  // namespace onnxruntime
