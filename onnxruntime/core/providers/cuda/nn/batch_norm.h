// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class BatchNorm final : public CudaKernel {
 public:
  BatchNorm(const OpKernelInfo& op_kernel_info)
      : CudaKernel{op_kernel_info},
        cudnn_batch_norm_mode_(CUDNN_BATCHNORM_SPATIAL),
        momentum_(0.9) {
    float tmp_epsilon;
    ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
    epsilon_ = ClampCudnnBatchNormEpsilon(static_cast<double>(tmp_epsilon));

    // spatial or not
    int64_t tmp_spatial;
    if (op_kernel_info.GetAttr<int64_t>("spatial", &tmp_spatial).IsOK()) {
      spatial_ = tmp_spatial;
    }

    if (spatial_ == 0) {
      cudnn_batch_norm_mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    }

    float tmp_momentum;
    if (op_kernel_info.GetAttr<float>("momentum", &tmp_momentum).IsOK()) {
      momentum_ = static_cast<double>(tmp_momentum);
    }

    is_training_mode_ = (op_kernel_info.GetAttrOrDefault<int64_t>("training_mode", 0) == 1);
    const auto& node = op_kernel_info.node();
    auto opset = node.SinceVersion();

    // batch norm opset 14 is not implemented for training mode
    ORT_ENFORCE(!(is_training_mode_ && opset==14), "Training mode does not support BN opset 14 yet.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  double epsilon_;
  int64_t spatial_ = 1;  // default as per spec
  cudnnBatchNormMode_t cudnn_batch_norm_mode_;
  double momentum_;
  bool is_training_mode_ = 0; //default as per spec
};

}  // namespace cuda
}  // namespace onnxruntime
