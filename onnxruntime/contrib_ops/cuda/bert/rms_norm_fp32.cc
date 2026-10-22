// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"

namespace onnxruntime::contrib::cuda {

class RMSNormFp32 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit RMSNormFp32(const OpKernelInfo& info) : CudaKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    ORT_ENFORCE(std::isfinite(epsilon_) && epsilon_ >= 0, "epsilon must be finite and nonnegative");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* x = context->Input<Tensor>(0);
    const auto* scale = context->Input<Tensor>(1);
    const auto& shape = x->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 2 && shape[1] == 3072 &&
                          shape.Size() <= std::numeric_limits<int>::max(),
                      "RMSNormFp32 X must have shape [rows,3072] with element count fitting int32");
    ORT_RETURN_IF_NOT(scale->Shape() == TensorShape({3072}),
                      "RMSNormFp32 scale must have shape [3072]");
    auto* output = context->Output(0, shape);
    if (shape[0] == 0) return Status::OK();
    onnxruntime::cuda::HostApplyRMSNormFp32(GetDeviceProp(), Stream(context),
                                            output->MutableDataRaw(), x->DataRaw(), scale->Data<float>(),
                                            static_cast<int>(shape[0]), 3072, epsilon_);
    return CUDA_CALL(cudaGetLastError());
  }

 private:
  float epsilon_;
};

ONNX_OPERATOR_KERNEL_EX(
    RMSNormFp32, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>()),
    RMSNormFp32);

}  // namespace onnxruntime::contrib::cuda
