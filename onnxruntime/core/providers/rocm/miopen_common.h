// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cfloat>

#include "core/providers/rocm/rocm_common.h"

#include <miopen/miopen.h>

const double MIOPEN_BN_MIN_EPSILON = 1e-5;

namespace onnxruntime {
namespace rocm {

#if MIOPEN_VERSION < 21800
typedef enum {
  miopenTensorNCHW = 0,
  miopenTensorNHWC = 1,
} miopenTensorLayout_t;
#endif

#define MIOPEN_CONVOLUTION_FWD_ALGO_COUNT 6
#define MIOPEN_CONVOLUTION_BWD_FILTER_ALGO_COUNT 4
#define MIOPEN_CONVOLUTION_BWD_DATA_ALGO_COUNT 6
#define MIOPEN_NCHW_LAYOUT miopenTensorNCHW
#define MIOPEN_NHWC_LAYOUT miopenTensorNHWC

class MiopenTensor final {
 public:
  MiopenTensor();
  ~MiopenTensor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MiopenTensor);

  Status Set(gsl::span<const int64_t> input_dims, miopenDataType_t dataType, bool is_nhwc = false);
  Status Set(miopenDataType_t dataType, miopenTensorLayout_t tensor_layout, int n, int c, int h, int w);
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

  Status Set(gsl::span<const int64_t> filter_dims, miopenDataType_t data_type);
  // Set 4D filter where k is output channels, c is input channels, h and w is rows and columns per filter.
  Status Set(miopenDataType_t data_type, miopenTensorLayout_t tensor_layout, int k, int c, int h, int w);

  operator miopenTensorDescriptor_t() const { return desc_; }

 private:
  miopenTensorDescriptor_t desc_;
};

template <typename ElemType>
struct Consts {
  static const constexpr ElemType Zero{0};
  static const constexpr ElemType One{1};
};

template <>
struct Consts<half> {
  static const constexpr float Zero{0};
  static const constexpr float One{1};
};

template <>
struct Consts<BFloat16> {
  static const constexpr float Zero{0};
  static const constexpr float One{1};
};

template <typename ElemType>
struct ReduceConsts {
  static const constexpr ElemType Zero{0};
  static const constexpr ElemType One{1};
};

#if ROCM_VERSION >= 40300
// Up until ROCm 4.2 miopenReduceTensor() required alpha/beta to be the same data
// type as the input type. This differs from cudnnReduceTensor() and other
// MIOpen/cuDNN APIs where alpha/beta are float when input type is half (float16).
template <>
struct ReduceConsts<half> {
  static const constexpr float Zero{0};
  static const constexpr float One{1};
};

template <>
struct ReduceConsts<BFloat16> {
  static const constexpr float Zero{0};
  static const constexpr float One{1};
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

inline miopenStatus_t
BatchNormalizationForwardInferenceHelper(miopenHandle_t handle,
                                         miopenBatchNormMode_t mode,
                                         const void* alpha,
                                         const void* beta,
                                         const miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         const miopenTensorDescriptor_t yDesc,
                                         void* y,
                                         const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                         const void* bnScale,
                                         const void* bnBias,
                                         const void* estimatedMean,
                                         const void* estimatedVariance,
                                         double epsilon) {
  return miopenBatchNormalizationForwardInference(handle,
                                                  mode,
                                                  const_cast<void*>(alpha),
                                                  const_cast<void*>(beta),
                                                  xDesc,
                                                  x,
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
BatchNormalizationForwardTrainingHelper(miopenHandle_t handle,
                                        miopenBatchNormMode_t mode,
                                        const void* alpha,
                                        const void* beta,
                                        const miopenTensorDescriptor_t xDesc,
                                        const void* x,
                                        const miopenTensorDescriptor_t yDesc,
                                        void* y,
                                        const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        const void* bnScale,
                                        const void* bnBias,
                                        double exponentialAverageFactor,
                                        void* resultRunningMean,
                                        void* resultRunningVariance,
                                        double epsilon,
                                        void* resultSaveMean,
                                        void* resultSaveInvVariance) {
  return miopenBatchNormalizationForwardTraining(handle,
                                                 mode,
                                                 const_cast<void*>(alpha),
                                                 const_cast<void*>(beta),
                                                 xDesc,
                                                 x,
                                                 yDesc,
                                                 y,
                                                 bnScaleBiasMeanVarDesc,
                                                 const_cast<void*>(bnScale),
                                                 const_cast<void*>(bnBias),
                                                 exponentialAverageFactor,
                                                 resultRunningMean,
                                                 resultRunningVariance,
                                                 epsilon,
                                                 resultSaveMean,
                                                 resultSaveInvVariance);
}

inline miopenStatus_t
LRNCrossChannelForwardHelper(miopenHandle_t handle,
                             miopenLRNDescriptor_t normDesc,
                             miopenLRNMode_t lrnMode,
                             const void* alpha,
                             const miopenTensorDescriptor_t xDesc,
                             const void* x,
                             const void* beta,
                             const miopenTensorDescriptor_t yDesc,
                             void* y) {
  if (lrnMode != miopenLRNCrossChannel) {
    LOGS_DEFAULT(ERROR) << __func__ << " must be called with lrnMode == miopenLRNCrossChannel";
    return miopenStatusBadParm;
  }
  return miopenLRNForward(handle, normDesc, alpha, xDesc, x, beta, yDesc, y, false, nullptr);
}

inline miopenStatus_t
SetLRNDescriptorHelper(miopenLRNDescriptor_t normDesc,
                       unsigned lrnN,
                       double lrnAlpha,
                       double lrnBeta,
                       double lrnK) {
  return miopenSetLRNDescriptor(normDesc, miopenLRNCrossChannel, lrnN, lrnAlpha, lrnBeta, lrnK);
}

inline miopenStatus_t
PoolingForwardHelper(miopenHandle_t handle,
                     const miopenPoolingDescriptor_t poolDesc,
                     const void* alpha,
                     const miopenTensorDescriptor_t xDesc,
                     const void* x,
                     const void* beta,
                     const miopenTensorDescriptor_t yDesc,
                     void* y) {
  return miopenPoolingForward(handle, poolDesc, alpha, xDesc, x, beta, yDesc, y, false, nullptr, 0);
}

inline miopenStatus_t
SetPoolingNdDescriptorHelper(miopenPoolingDescriptor_t poolDesc,
                             const miopenPoolingMode_t mode,
                             miopenNanPropagation_t /* unavailable */,
                             int nbDims,
                             int* windowDimA,
                             int* padA,
                             int* stridesA) {
  return miopenSetNdPoolingDescriptor(poolDesc, mode, nbDims, windowDimA, padA, stridesA);
}

}  // namespace rocm
}  // namespace onnxruntime
