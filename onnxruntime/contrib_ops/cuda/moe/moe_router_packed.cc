// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_router_packed.h"
#include "contrib_ops/cuda/moe/moe_router_packed_impl.h"

#include <cmath>
#include <limits>
#include <type_traits>

#include "core/common/safeint.h"

namespace onnxruntime::contrib::cuda {

#define REGISTER_MOE_ROUTER_PACKED_CUDA(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      MoERouterPacked, kMSDomain, 1, T, kCudaExecutionProvider,             \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<int32_t>())   \
          .TypeConstraint("Tweight", DataTypeImpl::GetTensorType<float>()), \
      MoERouterPacked<T>);

REGISTER_MOE_ROUTER_PACKED_CUDA(MLFloat16)
REGISTER_MOE_ROUTER_PACKED_CUDA(BFloat16)
#undef REGISTER_MOE_ROUTER_PACKED_CUDA

template <typename T>
MoERouterPacked<T>::MoERouterPacked(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("k", &k_).IsOK() && k_ > 0 && k_ <= 32,
              "MoERouterPacked requires k in 1..32");
  router_scale_ = info.GetAttrOrDefault<float>("router_scale", 1.0f);
  ORT_ENFORCE(std::isfinite(router_scale_), "MoERouterPacked requires finite router_scale");
}

template <typename T>
Status MoERouterPacked<T>::ComputeInternal(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto* weight = context->Input<Tensor>(1);
  const auto* bias = context->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(input && weight && bias, "MoERouterPacked requires input, weight and bias");
  ORT_RETURN_IF_NOT(input->Shape().NumDimensions() == 2 &&
                        weight->Shape().NumDimensions() == 2,
                    "MoERouterPacked input and weight must be rank two");
  const int64_t rows = input->Shape()[0];
  const int64_t width = input->Shape()[1];
  const int64_t experts = weight->Shape()[0];
  ORT_RETURN_IF_NOT(rows >= 0 && rows <= 2048 && width > 0 &&
                        width <= std::numeric_limits<int>::max() && width % 256 == 0 &&
                        experts >= k_ && experts <= 512 && weight->Shape()[1] == width &&
                        bias->Shape() == TensorShape({experts}),
                    "MoERouterPacked requires rows<=2048, width divisible by 256 and k<=experts<=512");
  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;
  ORT_RETURN_IF_NOT(!is_bf16 || GetDeviceProp().major >= 8,
                    "MoERouterPacked BF16 requires SM80+");
  auto* indices = context->Output(0, TensorShape({rows, k_}));
  auto* weights = context->Output(1, TensorShape({rows, k_}));
  if (rows == 0) return Status::OK();
  const size_t bytes = SafeInt<size_t>(rows) * experts * sizeof(float);
  auto scratch = GetScratchBuffer<uint8_t>(bytes, GetComputeStream(context));
  return LaunchMoERouterPackedCuda(
      Stream(context), is_bf16, static_cast<int>(rows), static_cast<int>(width),
      static_cast<int>(experts), static_cast<int>(k_), router_scale_,
      input->DataRaw(), weight->DataRaw(), bias->Data<float>(),
      reinterpret_cast<float*>(scratch.get()), indices->MutableData<int32_t>(),
      weights->MutableData<float>());
}

}  // namespace onnxruntime::contrib::cuda
