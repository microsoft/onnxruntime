// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/nn/causal_conv_with_state_impl.h"
#include "core/providers/causal_conv_with_state_helper.h"

namespace onnxruntime::cuda {

template <typename T>
class CausalConvWithState final : public CudaKernel {
 public:
  explicit CausalConvWithState(const OpKernelInfo& info)
      : CudaKernel(info),
        dilation_(info.GetAttrOrDefault<int64_t>("dilation", 1)),
        activation_(info.GetAttrOrDefault<std::string>("activation", "none")) {
    ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
                "CausalConvWithState activation '", activation_,
                "' is unsupported. Use 'none', 'silu', or 'swish'.");
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const auto* weight = context->Input<Tensor>(1);
    const auto* bias = context->Input<Tensor>(2);
    const auto* past = context->Input<Tensor>(3);
    CausalConvWithStateParameters p;
    ORT_RETURN_IF_ERROR(PrepareCausalConvWithState(*input, *weight, bias, past, dilation_, p));
    auto* output = context->Output(0, input->Shape());
    auto* present = context->Output(1, p.state_shape);
    using CudaT = typename ToCudaType<T>::MappedType;
    CausalConvWithStateImpl(
        Stream(context), reinterpret_cast<const CudaT*>(input->Data<T>()),
        reinterpret_cast<const CudaT*>(weight->Data<T>()),
        bias ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr,
        past ? reinterpret_cast<const CudaT*>(past->Data<T>()) : nullptr,
        reinterpret_cast<CudaT*>(output->MutableData<T>()),
        reinterpret_cast<CudaT*>(present->MutableData<T>()),
        SafeInt<int64_t>(p.batch) * p.channels, p.channels, p.length, p.kernel, dilation_, p.history,
        activation_ != "none");
    return Status::OK();
  }

 private:
  const int64_t dilation_;
  const std::string activation_;
};

#define REGISTER_CAUSAL_CONV29(T)                                                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(CausalConvWithState, kOnnxDomain, 29, T,                                             \
                                kCudaExecutionProvider,                                                              \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                CausalConvWithState<T>);

REGISTER_CAUSAL_CONV29(float)
REGISTER_CAUSAL_CONV29(MLFloat16)
REGISTER_CAUSAL_CONV29(BFloat16)
#undef REGISTER_CAUSAL_CONV29

}  // namespace onnxruntime::cuda
