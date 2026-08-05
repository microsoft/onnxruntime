// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <vector>

#include "contrib_ops/cpu/bert/hyper_connection_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {

template <typename T>
class HyperHead final : public OpKernel {
 public:
  explicit HyperHead(const OpKernelInfo& info) : OpKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    ORT_ENFORCE(epsilon_ > 0.0f, "HyperHead epsilon must be positive.");
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* hidden = context->Input<Tensor>(0);
    const Tensor* weight = context->Input<Tensor>(1);
    const Tensor* bias = context->Input<Tensor>(2);
    const Tensor* scale = context->Input<Tensor>(3);
    ORT_RETURN_IF_NOT(hidden && weight && bias && scale, "HyperHead requires all inputs.");
    int64_t rows, streams, hidden_size;
    ORT_RETURN_IF_ERROR(ValidateHyperParameters(*hidden, *weight, *bias, *scale, true, rows, streams, hidden_size));
    const int64_t flat_size = streams * hidden_size;
    const auto hidden_values = ToFloatVector<T>(*hidden);
    std::vector<float> normalized;
    std::vector<float> projected;
    NormalizeHyperInput(hidden_values, rows, flat_size, epsilon_, normalized);
    ProjectHyperInput(normalized, weight->Data<float>(), rows, flat_size, streams, projected);
    std::vector<float> output(static_cast<size_t>(rows * hidden_size), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t stream = 0; stream < streams; ++stream) {
        const float mix = 1.0f / (1.0f + std::exp(-(projected[static_cast<size_t>(row * streams + stream)] *
                                                   scale->Data<float>()[0] + bias->Data<float>()[stream]))) +
                          epsilon_;
        for (int64_t d = 0; d < hidden_size; ++d) {
          output[static_cast<size_t>(row * hidden_size + d)] +=
              mix * hidden_values[static_cast<size_t>((row * streams + stream) * hidden_size + d)];
        }
      }
    }
    TensorShapeVector output_shape{hidden->Shape()[0], hidden->Shape()[1], hidden_size};
    WriteFloatVector<T>(*context->Output(0, output_shape), output);
    return Status::OK();
  }

 private:
  float epsilon_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL(T)                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      HyperHead, kMSDomain, 1, T, kCpuExecutionProvider,              \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("F", DataTypeImpl::GetTensorType<float>()), \
      deepseek_v4_attention_impl::HyperHead<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime