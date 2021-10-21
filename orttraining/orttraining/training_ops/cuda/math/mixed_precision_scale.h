// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename SrcT, typename DstT>
void Impl_MixedPrecisionScale(
    cudaStream_t stream,
    const SrcT* input_data,
    const float* scale_data,
    DstT* output_data,
    size_t count);

template <typename SrcT>
class MixedPrecisionScale final : public CudaKernel {
 public:
  MixedPrecisionScale(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
  size_t bytes_per_output_elem_;
  bool fuse_outputs_;
};

}  // namespace cuda
}  // namespace onnxruntime
