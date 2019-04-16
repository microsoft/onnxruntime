// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/nn/conv_transpose.h"
#include "conv.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ConvTranspose : public CudaKernel, public ConvTransposeBase {
 public:
  ConvTranspose(const OpKernelInfo& info) : CudaKernel(info), ConvTransposeBase(info){};
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable CudnnConvState<cudnnConvolutionBwdDataAlgoPerf_t> s_;
};

}  // namespace cuda
}  // namespace onnxruntime
