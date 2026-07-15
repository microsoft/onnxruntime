// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/providers/cuda/cuda_common.h"
#include "core/common/narrow.h"
#include "skip_layer_norm.h"
#include "skip_layer_norm_impl.h"
#include "contrib_ops/cpu/skip_layer_norm_helper.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SkipLayerNormalization,                                     \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SkipLayerNorm<T, false>);                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      SkipSimplifiedLayerNormalization,                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SkipLayerNorm<T, true>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

using namespace ONNX_NAMESPACE;

template <typename T, bool Simplified>
SkipLayerNorm<T, Simplified>::SkipLayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(epsilon_ >= 0);
  // Note: the enable_skip_layer_norm_strict_mode provider option is deprecated and ignored.
  // The kernel always accumulates in fp32, so the previous strict-mode path is no longer needed.
}

template <typename T, bool Simplified>
Status SkipLayerNorm<T, Simplified>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* skip = ctx->Input<Tensor>(1);
  const Tensor* gamma = ctx->Input<Tensor>(2);

  const Tensor* beta = Simplified ? nullptr : ctx->Input<Tensor>(3);
  const Tensor* bias = Simplified ? ctx->Input<Tensor>(3) : ctx->Input<Tensor>(4);

  Tensor* output = ctx->Output(0, input->Shape());

  // Optional output for the sum of skip, input and bias tensors (It is also the input of Layer Normalization).
  Tensor* sum_output = ctx->Output(3, input->Shape());

  const auto& input_dims = input->Shape().GetDims();
  size_t input_dims_size = input_dims.size();

  int hidden_size = onnxruntime::narrow<int>(input_dims[input_dims_size - 1]);

  ORT_RETURN_IF_ERROR(onnxruntime::contrib::skip_layer_norm_helper::CheckInputs<Tensor>(input,
                                                                                        skip,
                                                                                        gamma,
                                                                                        beta,
                                                                                        bias,
                                                                                        hidden_size,
                                                                                        input_dims_size));

  int row_count = onnxruntime::narrow<int>(input->Shape().SizeToDimension(input_dims_size - 1));
  if (row_count == 0) {
    return Status::OK();
  }

  // Element offsets into the output are 32-bit on device; reject shapes whose element count would
  // exceed the 32-bit indexable range. The output shares the input shape, so input->Shape().Size()
  // is the output element count. The maximum output write index is
  // row_count * hidden_size - 1 == output element count - 1, so this guard covers every device write site.
  const int64_t output_element_count = input->Shape().Size();
  ORT_RETURN_IF_NOT(output_element_count <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                    "SkipLayerNormalization: output element count (", output_element_count,
                    ") exceeds the supported 32-bit indexing range.");

  typedef typename ToCudaType<T>::MappedType CudaT;

  const int skip_size = onnxruntime::narrow<int>(skip->Shape().Size());

  if constexpr (std::is_same_v<T, BFloat16>) {
    LaunchSkipLayerNormKernel<nv_bfloat16, Simplified>(
        Stream(ctx),
        reinterpret_cast<nv_bfloat16*>(output->MutableData<T>()),
        sum_output != nullptr ? reinterpret_cast<nv_bfloat16*>(sum_output->MutableData<T>()) : nullptr,
        reinterpret_cast<const nv_bfloat16*>(input->Data<T>()),
        reinterpret_cast<const nv_bfloat16*>(skip->Data<T>()),
        (bias != nullptr) ? reinterpret_cast<const nv_bfloat16*>(bias->Data<T>()) : nullptr,
        reinterpret_cast<const nv_bfloat16*>(gamma->Data<T>()),
        (beta != nullptr) ? reinterpret_cast<const nv_bfloat16*>(beta->Data<T>()) : nullptr,
        epsilon_,
        hidden_size,
        row_count,
        skip_size);
  } else {
    LaunchSkipLayerNormKernel<CudaT, Simplified>(
        Stream(ctx),
        reinterpret_cast<CudaT*>(output->MutableData<T>()),
        sum_output != nullptr ? reinterpret_cast<CudaT*>(sum_output->MutableData<T>()) : nullptr,
        reinterpret_cast<const CudaT*>(input->Data<T>()),
        reinterpret_cast<const CudaT*>(skip->Data<T>()),
        (bias != nullptr) ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr,
        reinterpret_cast<const CudaT*>(gamma->Data<T>()),
        (beta != nullptr) ? reinterpret_cast<const CudaT*>(beta->Data<T>()) : nullptr,
        epsilon_,
        hidden_size,
        row_count,
        skip_size);
  }

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
