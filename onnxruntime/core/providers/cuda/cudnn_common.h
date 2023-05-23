// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cfloat>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

class CudnnTensor final {
 public:
  CudnnTensor();
  ~CudnnTensor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudnnTensor);

  Status Set(gsl::span<const int64_t> input_dims, cudnnDataType_t dataType);
  Status Set(const CudnnTensor& x_desc, cudnnBatchNormMode_t mode);
  // Set 4D tensor format (for NHWC)
  Status Set(cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w);

  operator cudnnTensorDescriptor_t() const { return tensor_; }

  template <typename T>
  static cudnnDataType_t GetDataType();

 private:
  Status CreateTensorIfNeeded();

  cudnnTensorDescriptor_t tensor_;
};

class CudnnDataTensor final {
 public:
  CudnnDataTensor();
  ~CudnnDataTensor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudnnDataTensor);

  Status Set(cudnnDataType_t dataType,
             int64_t max_seq_length,
             int64_t batch_size,
             int64_t data_size,
             const int32_t* seq_lengths);

  operator cudnnRNNDataDescriptor_t() const { return tensor_; }

 private:
  Status CreateTensorIfNeeded();

  cudnnRNNDataDescriptor_t tensor_;
};

class CudnnFilterDescriptor final {
 public:
  CudnnFilterDescriptor();
  ~CudnnFilterDescriptor();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudnnFilterDescriptor);

  Status Set(gsl::span<const int64_t> filter_dims, cudnnDataType_t data_typ);

  // Set 4D filter where k is output channels, c is input channels, h and w is rows and columns per filter.
  Status Set(cudnnTensorFormat_t format, cudnnDataType_t dataType, int k, int c, int h, int w);

  operator cudnnFilterDescriptor_t() const { return desc_; }

 private:
  cudnnFilterDescriptor_t desc_;
};

class CudnnDropout final {
 public:
  CudnnDropout() : dropout_desc_(nullptr) {
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudnnDropout);

  Status GetCudnnDropoutStatesSize(const cudnnHandle_t& cudnnHandle, size_t& stateSize) {
    CUDNN_RETURN_IF_ERROR(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));

    return Status::OK();
  }

  Status Set(const cudnnHandle_t& cudnnHandle,
             void* states,
             size_t stateSize,
             float dropout = 0.0f,
             unsigned long long seed = 1) {
    ORT_RETURN_IF_ERROR(CreateDescriptorIfNeeded());
    CUDNN_RETURN_IF_ERROR(cudnnSetDropoutDescriptor(dropout_desc_,
                                                    cudnnHandle,
                                                    dropout,
                                                    states,
                                                    stateSize,
                                                    seed));

    return Status::OK();
  }

  ~CudnnDropout() {
    if (dropout_desc_ != nullptr) {
      cudnnDestroyDropoutDescriptor(dropout_desc_);
    }
  }

  operator cudnnDropoutDescriptor_t() const {
    return dropout_desc_;
  }

  Status CreateDescriptorIfNeeded() {
    if (!dropout_desc_)
      CUDNN_RETURN_IF_ERROR(cudnnCreateDropoutDescriptor(&dropout_desc_));
    return Status::OK();
  }

 private:
  cudnnDropoutDescriptor_t dropout_desc_;
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

inline double ClampCudnnBatchNormEpsilon(double epsilon) {
  if (epsilon < CUDNN_BN_MIN_EPSILON) {
    if (CUDNN_BN_MIN_EPSILON - epsilon > FLT_EPSILON)
      LOGS_DEFAULT(WARNING) << "Provided epsilon is smaller than CUDNN_BN_MIN_EPSILON. Setting it to CUDNN_BN_MIN_EPSILON";
    return CUDNN_BN_MIN_EPSILON;
  }
  return epsilon;
}

inline cudnnStatus_t
BatchNormalizationForwardInferenceHelper(cudnnHandle_t handle,
                                         cudnnBatchNormMode_t mode,
                                         const void* alpha,
                                         const void* beta,
                                         const cudnnTensorDescriptor_t xDesc,
                                         const void* x,
                                         const cudnnTensorDescriptor_t yDesc,
                                         void* y,
                                         const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                         const void* bnScale,
                                         const void* bnBias,
                                         const void* estimatedMean,
                                         const void* estimatedVariance,
                                         double epsilon) {
  return cudnnBatchNormalizationForwardInference(handle,
                                                 mode,
                                                 alpha,
                                                 beta,
                                                 xDesc,
                                                 x,
                                                 yDesc,
                                                 y,
                                                 bnScaleBiasMeanVarDesc,
                                                 bnScale,
                                                 bnBias,
                                                 estimatedMean,
                                                 estimatedVariance,
                                                 epsilon);
}

inline cudnnStatus_t
BatchNormalizationForwardTrainingHelper(cudnnHandle_t handle,
                                        cudnnBatchNormMode_t mode,
                                        const void* alpha,
                                        const void* beta,
                                        const cudnnTensorDescriptor_t xDesc,
                                        const void* x,
                                        const cudnnTensorDescriptor_t yDesc,
                                        void* y,
                                        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        const void* bnScale,
                                        const void* bnBias,
                                        double exponentialAverageFactor,
                                        void* resultRunningMean,
                                        void* resultRunningVariance,
                                        double epsilon,
                                        void* resultSaveMean,
                                        void* resultSaveInvVariance) {
  return cudnnBatchNormalizationForwardTraining(handle,
                                                mode,
                                                alpha,
                                                beta,
                                                xDesc,
                                                x,
                                                yDesc,
                                                y,
                                                bnScaleBiasMeanVarDesc,
                                                bnScale,
                                                bnBias,
                                                exponentialAverageFactor,
                                                resultRunningMean,
                                                resultRunningVariance,
                                                epsilon,
                                                resultSaveMean,
                                                resultSaveInvVariance);
}

inline cudnnStatus_t
LRNCrossChannelForwardHelper(cudnnHandle_t handle,
                             cudnnLRNDescriptor_t normDesc,
                             cudnnLRNMode_t lrnMode,
                             const void* alpha,
                             const cudnnTensorDescriptor_t xDesc,
                             const void* x,
                             const void* beta,
                             const cudnnTensorDescriptor_t yDesc,
                             void* y) {
  return cudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
}

inline cudnnStatus_t
SetLRNDescriptorHelper(cudnnLRNDescriptor_t normDesc,
                       unsigned lrnN,
                       double lrnAlpha,
                       double lrnBeta,
                       double lrnK) {
  return cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
}

inline cudnnStatus_t
PoolingForwardHelper(cudnnHandle_t handle,
                     const cudnnPoolingDescriptor_t poolingDesc,
                     const void* alpha,
                     const cudnnTensorDescriptor_t xDesc,
                     const void* x,
                     const void* beta,
                     const cudnnTensorDescriptor_t yDesc,
                     void* y) {
  return cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
}

inline cudnnStatus_t
SetPoolingNdDescriptorHelper(cudnnPoolingDescriptor_t poolingDesc,
                             const cudnnPoolingMode_t mode,
                             const cudnnNanPropagation_t maxpoolingNanOpt,
                             int nbDims,
                             const int windowDimA[],
                             const int paddingA[],
                             const int strideA[]) {
  return cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
}

}  // namespace cuda
}  // namespace onnxruntime
