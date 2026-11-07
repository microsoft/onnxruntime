// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <string>

#include "contrib_ops/cuda/quantization/moe_packed_impl.cuh"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class MoEPacked final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MoEPacked(const OpKernelInfo& info) : CudaKernel(info) {
    std::string quant_type;
    ORT_THROW_IF_ERROR(info.GetAttr<std::string>("quant_type", &quant_type));
    ORT_ENFORCE(quant_type == "iq2_xxs", "MoEPacked requires quant_type='iq2_xxs'.");
    ORT_THROW_IF_ERROR(info.GetAttr<int64_t>("k", &topk_));
    ORT_ENFORCE(topk_ >= 1 && topk_ <= 32, "MoEPacked requires 1 <= k <= 32.");
    router_scale_ = info.GetAttrOrDefault<float>("router_scale", 1.0f);
    ORT_ENFORCE(std::isfinite(router_scale_), "MoEPacked router_scale must be finite.");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* x = context->Input<Tensor>(0);
    const auto* fc1 = context->Input<Tensor>(1);
    const auto* fc2 = context->Input<Tensor>(2);
    const auto optional_input = [context](int index) -> const Tensor* {
      return context->InputCount() > index ? context->Input<Tensor>(index) : nullptr;
    };
    const auto* indices = optional_input(3);
    const auto* weights = optional_input(4);
    const auto* router_input = optional_input(5);
    const auto* router_weight = optional_input(6);
    const auto* router_bias = optional_input(7);
    const bool fused_router = router_input || router_weight || router_bias;
    ORT_RETURN_IF_NOT(GetDeviceProp().major >= 8, "MoEPacked requires CUDA SM80 or later.");
    ORT_RETURN_IF_NOT(x->Shape().NumDimensions() == 2 && x->Shape()[1] == 1024 &&
                          x->Shape()[0] >= 1 && x->Shape()[0] <= 8 && x->Shape()[0] * topk_ <= 128,
                      "MoEPacked requires X[R,1024], 1<=R<=8 and R*k<=128.");
    const int rows = static_cast<int>(x->Shape()[0]), topk = static_cast<int>(topk_);
    const int pairs = rows * topk;
    ORT_RETURN_IF_NOT(fc1->Shape().NumDimensions() == 3 &&
                          fc1->Shape()[0] >= topk && fc1->Shape()[0] <= std::numeric_limits<int32_t>::max() &&
                          fc1->Shape()[1] > 0 && fc1->Shape()[1] <= std::numeric_limits<int32_t>::max() &&
                          fc1->Shape()[1] % 256 == 0 && fc1->Shape()[2] == 264,
                      "MoEPacked requires FC1[E,I,264], E>=k and positive int32 I divisible by 256.");
    const int experts = static_cast<int>(fc1->Shape()[0]);
    const int intermediate = static_cast<int>(fc1->Shape()[1]);
    ORT_RETURN_IF_NOT(fc2->Shape() == TensorShape({experts, 1024, static_cast<int64_t>(intermediate / 256) * 66}),
                      "MoEPacked requires FC2[E,1024,(I/256)*66].");
    IAllocatorUniquePtr<int32_t> routed_indices;
    IAllocatorUniquePtr<float> routed_weights, router_scores;
    if (fused_router) {
      ORT_RETURN_IF_NOT(router_input && router_weight && router_bias && !indices && !weights &&
                            router_input->Shape().NumDimensions() == 2 &&
                            router_input->Shape()[0] == rows && router_input->Shape()[1] > 0 &&
                            router_input->Shape()[1] <= std::numeric_limits<int32_t>::max() &&
                            router_input->Shape()[1] % 256 == 0 && experts <= 512,
                        "MoEPacked fused routing requires router_input[R,H], positive int32 H "
                        "divisible by 256, E<=512 and no preselected routes.");
      const int width = static_cast<int>(router_input->Shape()[1]);
      ORT_RETURN_IF_NOT(router_weight->Shape() == TensorShape({experts, width}) &&
                            router_bias->Shape() == TensorShape({experts}),
                        "MoEPacked requires router_weight[E,H] and router_bias[E].");
      routed_indices = GetScratchBuffer<int32_t>(pairs, GetComputeStream(context));
      routed_weights = GetScratchBuffer<float>(pairs, GetComputeStream(context));
      router_scores = GetScratchBuffer<float>(SafeInt<size_t>(rows) * experts, GetComputeStream(context));
      CUDA_RETURN_IF_ERROR(LaunchMoEPackedRouter(
          Stream(context), rows, width, experts, topk, router_scale_,
          router_input->DataRaw(), router_weight->DataRaw(), router_bias->Data<float>(),
          routed_indices.get(), routed_weights.get(), router_scores.get()));
    } else {
      ORT_RETURN_IF_NOT(indices && weights && indices->Shape() == TensorShape({rows, topk}) &&
                            weights->Shape() == indices->Shape(),
                        "MoEPacked preselected routes require indices[R,k] and weights[R,k].");
      ORT_RETURN_IF_NOT(router_scale_ == 1.0f, "MoEPacked non-default router_scale requires fused routing.");
    }
    auto hidden = GetScratchBuffer<BFloat16>(SafeInt<size_t>(pairs) * intermediate, GetComputeStream(context));
    auto partial = GetScratchBuffer<float>(SafeInt<size_t>(pairs) * 1024, GetComputeStream(context));
    auto metadata = GetScratchBuffer<int32_t>(1 + pairs * 9, GetComputeStream(context));
    auto* y = context->Output(0, x->Shape());
    CUDA_RETURN_IF_ERROR(LaunchMoEPacked(
        Stream(context), rows, topk, experts, intermediate, x->DataRaw(),
        fused_router ? routed_indices.get() : indices->Data<int32_t>(),
        fused_router ? routed_weights.get() : weights->Data<float>(),
        fc1->Data<uint8_t>(), fc2->Data<uint8_t>(), y->MutableDataRaw(),
        hidden.get(), partial.get(), metadata.get()));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MoEPacked);
  int64_t topk_;
  float router_scale_;
};

ONNX_OPERATOR_KERNEL_EX(
    MoEPacked, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("TW", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("TF", DataTypeImpl::GetTensorType<float>()),
    MoEPacked);

}  // namespace onnxruntime::contrib::cuda
