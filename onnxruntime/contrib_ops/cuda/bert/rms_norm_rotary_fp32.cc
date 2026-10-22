// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"

namespace onnxruntime::contrib::cuda {

class RMSNormRotaryFp32 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit RMSNormRotaryFp32(const OpKernelInfo& info) : CudaKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    ORT_ENFORCE(std::isfinite(epsilon_) && epsilon_ >= 0, "epsilon must be finite and nonnegative");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* x = context->Input<Tensor>(0);
    const auto* scale = context->Input<Tensor>(1);
    const auto* cos = context->Input<Tensor>(2);
    const auto* sin = context->Input<Tensor>(3);
    const auto& shape = x->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 3 && shape[2] == 256 &&
                          (shape[1] == 2 || shape[1] == 32) &&
                          shape.Size() <= std::numeric_limits<int>::max(),
                      "RMSNormRotaryFp32 X must be [tokens,2|32,256], elements fitting int32");
    ORT_RETURN_IF_NOT(scale->Shape() == TensorShape({256}) &&
                          cos->Shape() == TensorShape({1, shape[0], 32}) &&
                          sin->Shape() == cos->Shape(),
                      "RMSNormRotaryFp32 scale must be [256] and cos/sin [1,tokens,32]");
    auto* output = context->Output(0, TensorShape({1, shape[0], shape[1] * 256}));
    if (shape[0] == 0) return Status::OK();
    onnxruntime::cuda::HostApplyRMSNormRotaryFp32(
        GetDeviceProp(), Stream(context), output->MutableDataRaw(), x->DataRaw(), scale->Data<float>(),
        cos->DataRaw(), sin->DataRaw(), static_cast<int>(shape[0]), static_cast<int>(shape[1]), epsilon_);
    return CUDA_CALL(cudaGetLastError());
  }

 private:
  float epsilon_;
};

ONNX_OPERATOR_KERNEL_EX(
    RMSNormRotaryFp32, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>()),
    RMSNormRotaryFp32);

}  // namespace onnxruntime::contrib::cuda
