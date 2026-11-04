// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/narrow.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/quantization/packed_moe_impl.h"

namespace onnxruntime::contrib::cuda {
class MoEPackedIQ2XXS final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MoEPackedIQ2XXS(const OpKernelInfo& info) : CudaKernel(info) {
    topk_ = narrow<int>(info.GetAttrOrDefault<int64_t>("topk", 0));
    ORT_ENFORCE(topk_ > 0 && topk_ <= 32);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MoEPackedIQ2XXS);
  Status ComputeInternal(OpKernelContext* context) const override {
    ORT_RETURN_IF_NOT(GetDeviceProp().major >= 8, "MoEPackedIQ2XXS requires SM80 or later");
    const auto* x = context->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(x->Shape().NumDimensions() == 2 && x->Shape()[1] == 1024, "Expected X[R,1024]");
    const int64_t rows64 = x->Shape()[0];
    ORT_RETURN_IF_NOT(rows64 >= 1 && rows64 <= 8 && rows64 * topk_ <= 128, "Unsupported row or pair count");
    const int rows = static_cast<int>(rows64), pairs = rows * topk_;
    const auto* indices = context->Input<Tensor>(1);
    const auto* weights = context->Input<Tensor>(2);
    ORT_RETURN_IF_NOT(indices->Shape() == TensorShape({rows, topk_}) &&
                          weights->Shape() == indices->Shape(),
                      "Expected routing matrices [R,topk]");
    const auto* fc1 = context->Input<Tensor>(3);
    const auto* scale1 = context->Input<Tensor>(4);
    const auto* fc2 = context->Input<Tensor>(5);
    const auto* scale2 = context->Input<Tensor>(6);
    ORT_RETURN_IF_NOT(fc1->Shape().NumDimensions() == 4, "Expected FC1[E,I,32,8]");
    const int experts = narrow<int>(fc1->Shape()[0]);
    const int intermediate = narrow<int>(fc1->Shape()[1]);
    ORT_RETURN_IF_NOT(experts >= topk_ && intermediate > 0 && intermediate % 256 == 0 &&
                          fc1->Shape()[2] == 32 && fc1->Shape()[3] == 8 &&
                          scale1->Shape() == TensorShape({experts, intermediate, 4}) &&
                          fc2->Shape() == TensorShape({experts, 1024, intermediate / 32, 8}) &&
                          scale2->Shape() == TensorShape({experts, 1024, intermediate / 256}),
                      "Invalid packed IQ2 layout");
    auto* y = context->Output(0, {rows, 1024});
    auto* hidden = context->Output(1, {pairs, intermediate});
    auto* expert_output = context->Output(2, {pairs, 1024});
    auto* metadata = context->Output(3, {1 + pairs * 9});
    CUDA_RETURN_IF_ERROR(LaunchMoEPackedIQ2XXS(
        Stream(context), rows, topk_, intermediate, x->DataRaw(), indices->Data<int32_t>(), weights->Data<float>(),
        fc1->Data<uint8_t>(), scale1->DataRaw(), fc2->Data<uint8_t>(), scale2->DataRaw(), y->MutableDataRaw(),
        hidden->MutableDataRaw(), expert_output->MutableData<float>(), metadata->MutableData<int32_t>()));
    return Status::OK();
  }

 private:
  int topk_;
};

ONNX_OPERATOR_KERNEL_EX(
    MoEPackedIQ2XXS, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()).TypeConstraint("S", {DataTypeImpl::GetTensorType<BFloat16>(), DataTypeImpl::GetTensorType<MLFloat16>()}).TypeConstraint("F", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int32_t>()).TypeConstraint("Q", DataTypeImpl::GetTensorType<uint8_t>()),
    MoEPackedIQ2XXS);
}  // namespace onnxruntime::contrib::cuda
