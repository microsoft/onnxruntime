// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/narrow.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/sequence/fused_state_space_impl.h"

#include <cmath>
#include <limits>

namespace onnxruntime::contrib::cuda {

using onnxruntime::cuda::CudaKernel;

class Mamba2Verify final : public CudaKernel {
 public:
  explicit Mamba2Verify(const OpKernelInfo& info) : CudaKernel(info) {
    heads_ = narrow<int>(info.GetAttrOrDefault<int64_t>("heads", 0));
    groups_ = narrow<int>(info.GetAttrOrDefault<int64_t>("n_groups", 0));
    ORT_ENFORCE(heads_ > 0 && groups_ > 0 && heads_ % groups_ == 0 &&
                heads_ <= std::numeric_limits<int>::max() / 1024);
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("head_dim", 64) == 64 &&
                    info.GetAttrOrDefault<int64_t>("d_state", 128) == 128,
                "Mamba2Verify currently requires head_dim=64, d_state=128");
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
    time_step_min_ = info.GetAttrOrDefault<float>("time_step_min", 0.001f);
    ORT_ENFORCE(std::isfinite(epsilon_) && epsilon_ > 0 &&
                std::isfinite(time_step_min_) && time_step_min_ >= 0);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Mamba2Verify);

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* projection = context->Input<Tensor>(0);
    const auto* conv = context->Input<Tensor>(1);
    const auto* ssm = context->Input<Tensor>(2);
    const int di = heads_ * 64;
    const int channels = di + 2 * groups_ * 128;
    ORT_RETURN_IF_NOT(projection->Shape().NumDimensions() == 2, "projection must have rank 2");
    const int64_t rows64 = projection->Shape()[0];
    ORT_RETURN_IF_NOT(rows64 >= 1 && rows64 <= 16 &&
                          projection->Shape()[1] == di + channels + heads_,
                      "Invalid projection shape");
    const int rows = static_cast<int>(rows64);
    ORT_RETURN_IF_NOT(conv->Shape().NumDimensions() == 3 && conv->Shape()[0] >= rows &&
                          conv->Shape()[1] == channels && conv->Shape()[2] == 3,
                      "Invalid convolution state shape");
    const int64_t slots = conv->Shape()[0];
    ORT_RETURN_IF_NOT(ssm->Shape() == TensorShape({slots, heads_, 128, 64}), "Invalid SSM state shape");
    const auto& conv_weight_shape = context->Input<Tensor>(3)->Shape();
    ORT_RETURN_IF_NOT((conv_weight_shape == TensorShape({channels, 4}) ||
                       conv_weight_shape == TensorShape({channels, 1, 4})) &&
                          context->Input<Tensor>(4)->Shape() == TensorShape({channels}),
                      "Invalid convolution parameters");
    for (int i = 5; i <= 7; ++i) {
      ORT_RETURN_IF_NOT(context->Input<Tensor>(i)->Shape() == TensorShape({heads_}), "Invalid head parameter");
    }
    ORT_RETURN_IF_NOT(context->Input<Tensor>(8)->Shape() == TensorShape({di}) &&
                          context->Input<Tensor>(9)->Shape() == TensorShape({1}),
                      "Invalid norm weight or state_row");
    auto* y = context->Output(0, {rows, di});
    auto* conv_out = context->Output(1, conv->Shape());
    auto* ssm_out = context->Output(2, ssm->Shape());
    auto* xbc = context->Output(3, {rows, channels});
    auto* readout = context->Output(4, {rows, di});
    // Unwritten snapshot slots retain their input values, including when ORT does not alias buffers.
    if (conv_out->MutableDataRaw() != conv->DataRaw()) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(conv_out->MutableDataRaw(), conv->DataRaw(), conv->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream(context)));
    }
    if (ssm_out->MutableDataRaw() != ssm->DataRaw()) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(ssm_out->MutableDataRaw(), ssm->DataRaw(), ssm->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream(context)));
    }
    CUDA_RETURN_IF_ERROR(LaunchMamba2Verify(
        Stream(context), rows, slots, heads_, groups_, epsilon_, time_step_min_, projection->DataRaw(), conv->DataRaw(),
        ssm->Data<float>(), context->Input<Tensor>(3)->DataRaw(), context->Input<Tensor>(4)->DataRaw(),
        context->Input<Tensor>(5)->Data<float>(), context->Input<Tensor>(6)->Data<float>(),
        context->Input<Tensor>(7)->Data<float>(), context->Input<Tensor>(8)->Data<float>(),
        context->Input<Tensor>(9)->Data<int32_t>(), y->MutableDataRaw(), conv_out->MutableDataRaw(),
        ssm_out->MutableData<float>(), xbc->MutableDataRaw(), readout->MutableData<float>()));
    return Status::OK();
  }

 private:
  int heads_, groups_;
  float epsilon_, time_step_min_;
};

class BiasCorrectedMoERoute final : public CudaKernel {
 public:
  explicit BiasCorrectedMoERoute(const OpKernelInfo& info) : CudaKernel(info) {
    topk_ = narrow<int>(info.GetAttrOrDefault<int64_t>("topk", 0));
    scale_ = info.GetAttrOrDefault<float>("scale", 1.0f);
    ORT_ENFORCE(topk_ > 0 && topk_ <= 32 && std::isfinite(scale_));
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BiasCorrectedMoERoute);
  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* x = context->Input<Tensor>(0);
    const auto* weight = context->Input<Tensor>(1);
    const auto* bias = context->Input<Tensor>(2);
    ORT_RETURN_IF_NOT(x->Shape().NumDimensions() == 2 && weight->Shape().NumDimensions() == 2, "Expected matrices");
    const int rows = narrow<int>(x->Shape()[0]);
    const int width = narrow<int>(x->Shape()[1]);
    const int experts = narrow<int>(weight->Shape()[0]);
    ORT_RETURN_IF_NOT(rows > 0 && width > 0 && width % 256 == 0 && experts >= topk_ && experts <= 512 &&
                          weight->Shape()[1] == width && bias->Shape() == TensorShape({experts}),
                      "Unsupported router shape");
    auto* indices = context->Output(0, {rows, topk_});
    auto* weights = context->Output(1, {rows, topk_});
    auto* scores = context->Output(2, {rows, experts});
    CUDA_RETURN_IF_ERROR(LaunchBiasCorrectedMoERoute(
        Stream(context), rows, width, experts, topk_, scale_, x->DataRaw(), weight->DataRaw(), bias->Data<float>(),
        indices->MutableData<int32_t>(), weights->MutableData<float>(), scores->MutableData<float>()));
    return Status::OK();
  }

 private:
  int topk_;
  float scale_;
};

class ReluSquared final : public CudaKernel {
 public:
  explicit ReluSquared(const OpKernelInfo& info) : CudaKernel(info) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReluSquared);
  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* x = context->Input<Tensor>(0);
    auto* y = context->Output(0, x->Shape());
    CUDA_RETURN_IF_ERROR(LaunchReluSquared(Stream(context), x->DataRaw(), y->MutableDataRaw(), x->Shape().Size()));
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    Mamba2Verify, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()).TypeConstraint("F", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int32_t>()).MayInplace(1, 1).MayInplace(2, 2),
    Mamba2Verify);
ONNX_OPERATOR_KERNEL_EX(
    BiasCorrectedMoERoute, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()).TypeConstraint("F", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int32_t>()),
    BiasCorrectedMoERoute);
ONNX_OPERATOR_KERNEL_EX(
    ReluSquared, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()), ReluSquared);

}  // namespace onnxruntime::contrib::cuda
