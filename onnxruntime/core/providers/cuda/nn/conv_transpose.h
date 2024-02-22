// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cpu/nn/conv_transpose_attributes.h"

namespace onnxruntime {
namespace cuda {

template <typename T, bool NHWC>
class ConvTranspose : public CudaKernel {
 public:
  ConvTranspose(const OpKernelInfo& info) : CudaKernel(info), conv_transpose_attrs_(info){};
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, [[maybe_unused]] PrePackedWeights* prepacked_weights) override;
  Status ComputeInternal(OpKernelContext* context) const override;
  Status DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const;

 private:
  ConvTransposeAttributes conv_transpose_attrs_;

  mutable CudnnConvState<cudnnConvolutionBwdDataAlgoPerf_t> s_;
  std::unique_ptr<Tensor> W_;
};

}  // namespace cuda
}  // namespace onnxruntime
