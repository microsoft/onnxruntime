// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/deepseek_v4_compression_common.h"
#include "contrib_ops/cuda/bert/hyper_connection.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

static Status ValidateHyperInputs(const Tensor& hidden, const Tensor& weight, const Tensor& bias,
                                  const Tensor& scale, bool is_head, int64_t& rows,
                                  int64_t& streams, int64_t& hidden_size) {
  ORT_RETURN_IF_NOT(hidden.Shape().NumDimensions() == 4 && weight.Shape().NumDimensions() == 2 &&
                        bias.Shape().NumDimensions() == 1 && scale.Shape().NumDimensions() == 1,
                    "HyperConnection inputs have invalid ranks.");
  streams = hidden.Shape()[2];
  hidden_size = hidden.Shape()[3];
  rows = hidden.Shape()[0] * hidden.Shape()[1];
  const int64_t output_size = is_head ? streams : (2 + streams) * streams;
  ORT_RETURN_IF_NOT(streams > 0 && hidden_size > 0 && weight.Shape()[0] == output_size &&
                        weight.Shape()[1] == streams * hidden_size && bias.Shape().Size() == output_size &&
                        scale.Shape().Size() == (is_head ? 1 : 3) && weight.IsDataType<float>() &&
                        bias.IsDataType<float>() && scale.IsDataType<float>(),
                    "HyperConnection parameter shapes or types are inconsistent with hidden_streams.");
  return Status::OK();
}

template <typename T>
class HyperConnection final : public CudaKernel {
 public:
  explicit HyperConnection(const OpKernelInfo& info) : CudaKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    sinkhorn_iterations_ = info.GetAttrOrDefault<int64_t>("sinkhorn_iterations", 20);
    ORT_ENFORCE(epsilon_ > 0.0f && sinkhorn_iterations_ > 0);
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* hidden = context->Input<Tensor>(0);
    const Tensor* weight = context->Input<Tensor>(1);
    const Tensor* bias = context->Input<Tensor>(2);
    const Tensor* scale = context->Input<Tensor>(3);
    ORT_RETURN_IF_NOT(hidden && weight && bias && scale, "HyperConnection requires all inputs.");
    int64_t rows, streams, hidden_size;
    ORT_RETURN_IF_ERROR(ValidateHyperInputs(*hidden, *weight, *bias, *scale, false, rows, streams, hidden_size));
    TensorShapeVector post_shape{hidden->Shape()[0], hidden->Shape()[1], streams};
    TensorShapeVector comb_shape{hidden->Shape()[0], hidden->Shape()[1], streams, streams};
    TensorShapeVector collapsed_shape{hidden->Shape()[0], hidden->Shape()[1], hidden_size};
    Tensor* post = context->Output(0, post_shape);
    Tensor* comb = context->Output(1, comb_shape);
    Tensor* collapsed = context->Output(2, collapsed_shape);
    using CudaT = typename ToCudaType<T>::MappedType;
    return LaunchHyperConnectionKernel<CudaT>(
        Stream(context), reinterpret_cast<CudaT*>(post->MutableData<T>()),
        reinterpret_cast<CudaT*>(comb->MutableData<T>()),
        reinterpret_cast<CudaT*>(collapsed->MutableData<T>()),
        reinterpret_cast<const CudaT*>(hidden->Data<T>()), weight->Data<float>(),
        bias->Data<float>(), scale->Data<float>(), narrow<int>(rows), narrow<int>(streams),
        narrow<int>(hidden_size), epsilon_, narrow<int>(sinkhorn_iterations_),
        GetDeviceProp().maxThreadsPerBlock);
  }

 private:
  float epsilon_{};
  int64_t sinkhorn_iterations_{};
};

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

#define REGISTER_HYPER_KERNEL(OP, T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      OP, kMSDomain, 1, T, kCudaExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("F", DataTypeImpl::GetTensorType<float>()), \
      OP<T>);

REGISTER_HYPER_KERNEL(HyperConnection, float)
REGISTER_HYPER_KERNEL(HyperConnection, MLFloat16)
REGISTER_HYPER_KERNEL(HyperConnection, BFloat16)
REGISTER_HYPER_KERNEL(HyperHead, float)
REGISTER_HYPER_KERNEL(HyperHead, MLFloat16)
REGISTER_HYPER_KERNEL(HyperHead, BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime