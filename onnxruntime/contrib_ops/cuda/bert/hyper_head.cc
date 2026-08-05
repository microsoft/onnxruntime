// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/deepseek_v4_compression_common.h"
#include "contrib_ops/cuda/bert/hyper_connection_common.h"
#include "contrib_ops/cuda/bert/hyper_head.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

template <typename T>
class HyperHead final : public CudaKernel {
 public:
  explicit HyperHead(const OpKernelInfo& info) : CudaKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    ORT_ENFORCE(epsilon_ > 0.0f);
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* hidden = context->Input<Tensor>(0);
    const Tensor* weight = context->Input<Tensor>(1);
    const Tensor* bias = context->Input<Tensor>(2);
    const Tensor* scale = context->Input<Tensor>(3);
    ORT_RETURN_IF_NOT(hidden && weight && bias && scale, "HyperHead requires all inputs.");
    int64_t rows, streams, hidden_size;
    ORT_RETURN_IF_ERROR(ValidateHyperInputs(*hidden, *weight, *bias, *scale, true, rows, streams, hidden_size));
    TensorShapeVector output_shape{hidden->Shape()[0], hidden->Shape()[1], hidden_size};
    Tensor* output = context->Output(0, output_shape);
    using CudaT = typename ToCudaType<T>::MappedType;
    return LaunchHyperHeadKernel<CudaT>(
        Stream(context), reinterpret_cast<CudaT*>(output->MutableData<T>()),
        reinterpret_cast<const CudaT*>(hidden->Data<T>()), weight->Data<float>(),
        bias->Data<float>(), scale->Data<float>(), narrow<int>(rows), narrow<int>(streams),
        narrow<int>(hidden_size), epsilon_, GetDeviceProp().maxThreadsPerBlock);
  }

 private:
  float epsilon_{};
};

}  // namespace

#define REGISTER_KERNEL(T)                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      HyperHead, kMSDomain, 1, T, kCudaExecutionProvider,             \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("F", DataTypeImpl::GetTensorType<float>()), \
      HyperHead<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)
REGISTER_KERNEL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime