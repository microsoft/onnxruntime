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
  using CudaT = typename ToCudaType<T>::MappedType;

  ConvTranspose(const OpKernelInfo& info) : CudaKernel(info), conv_transpose_attrs_(info) {};
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, [[maybe_unused]] PrePackedWeights* prepacked_weights) override;
  Status ComputeInternal(OpKernelContext* context) const override;
  Status DoConvTranspose(OpKernelContext* context, bool dynamic_padding) const;

 private:
  ConvTransposeAttributes conv_transpose_attrs_;

  mutable CudnnConvState<cudnnConvolutionBwdDataAlgoPerf_t> s_;
  std::unique_ptr<Tensor> W_;

  bool is_nhwc_domain_;         // prepack is only needed for the Conv in kMSInternalNHWCDomain
  bool is_fused_node_ = false;  // ensures the node is fused although the session option is not set
  bool W_already_nhwc = false;  // In case NHWC == true and Conv is not in kMSInternalNHWCDomain

 protected:
  inline IAllocatorUniquePtr<void> GetWorkSpace(onnxruntime::Stream* stream) const {
    return GetScratchBuffer<void>(s_.workspace_bytes, stream);
  }

  Status UpdateState(OpKernelContext* context, bool bias_expected) const;

#if !defined(__CUDACC__) && CUDNN_MAJOR >= 9
  Status CreateCudnnFeExecutionPlan(const onnxruntime::TensorShapeVector& x_dims,
                                    const onnxruntime::TensorShapeVector& w_dims,
                                    const Tensor* B,
                                    const TensorShapeVector& y_dims,
                                    cudnnContext* handle,
                                    const cudnn_frontend::HeurMode_t heur_mode,
                                    const std::vector<int64_t>& pads,
                                    const std::vector<int64_t>& strides,
                                    const std::vector<int64_t>& dilations,
                                    const bool fuse_bias,
                                    const bool fuse_act,
                                    const bool w_in_nhwc,
                                    const bool use_tf32) const;
#endif
};

}  // namespace cuda
}  // namespace onnxruntime
