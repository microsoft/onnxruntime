// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <vector>

#include "contrib_ops/cpu/bert/hyper_connection_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {

template <typename T>
class HyperConnection final : public OpKernel {
 public:
  explicit HyperConnection(const OpKernelInfo& info) : OpKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    sinkhorn_iterations_ = info.GetAttrOrDefault<int64_t>("sinkhorn_iterations", 20);
    ORT_ENFORCE(epsilon_ > 0.0f && sinkhorn_iterations_ > 0,
                "HyperConnection epsilon and sinkhorn_iterations must be positive.");
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* hidden = context->Input<Tensor>(0);
    const Tensor* weight = context->Input<Tensor>(1);
    const Tensor* bias = context->Input<Tensor>(2);
    const Tensor* scale = context->Input<Tensor>(3);
    ORT_RETURN_IF_NOT(hidden && weight && bias && scale, "HyperConnection requires all inputs.");
    int64_t rows, streams, hidden_size;
    ORT_RETURN_IF_ERROR(ValidateHyperParameters(*hidden, *weight, *bias, *scale, false, rows, streams, hidden_size));

    const int64_t flat_size = streams * hidden_size;
    const int64_t projection_size = (2 + streams) * streams;
    const auto hidden_values = ToFloatVector<T>(*hidden);
    std::vector<float> normalized;
    std::vector<float> projected;
    NormalizeHyperInput(hidden_values, rows, flat_size, epsilon_, normalized);
    ProjectHyperInput(normalized, weight->Data<float>(), rows, flat_size, projection_size, projected);

    std::vector<float> pre(static_cast<size_t>(rows * streams));
    std::vector<float> post(pre.size());
    std::vector<float> comb(static_cast<size_t>(rows * streams * streams));
    const float* scales = scale->Data<float>();
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t stream = 0; stream < streams; ++stream) {
        pre[static_cast<size_t>(row * streams + stream)] =
            1.0f / (1.0f + std::exp(-(projected[static_cast<size_t>(row * projection_size + stream)] * scales[0] +
                          bias->Data<float>()[stream]))) + epsilon_;
        post[static_cast<size_t>(row * streams + stream)] =
            2.0f / (1.0f + std::exp(-(projected[static_cast<size_t>(row * projection_size + streams + stream)] * scales[1] +
                          bias->Data<float>()[streams + stream])));
      }
      const int64_t comb_offset = row * streams * streams;
      const int64_t projection_offset = row * projection_size + 2 * streams;
      for (int64_t output_stream = 0; output_stream < streams; ++output_stream) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int64_t input_stream = 0; input_stream < streams; ++input_stream) {
          maximum = std::max(maximum, projected[static_cast<size_t>(projection_offset + output_stream * streams + input_stream)] * scales[2] +
                                          bias->Data<float>()[2 * streams + output_stream * streams + input_stream]);
        }
        float denominator = 0.0f;
        for (int64_t input_stream = 0; input_stream < streams; ++input_stream) {
          const size_t index = static_cast<size_t>(comb_offset + output_stream * streams + input_stream);
          comb[index] = std::exp(projected[static_cast<size_t>(projection_offset + output_stream * streams + input_stream)] * scales[2] +
                                 bias->Data<float>()[2 * streams + output_stream * streams + input_stream] - maximum);
          denominator += comb[index];
        }
        for (int64_t input_stream = 0; input_stream < streams; ++input_stream) {
          const size_t index = static_cast<size_t>(comb_offset + output_stream * streams + input_stream);
          comb[index] = comb[index] / denominator + epsilon_;
        }
      }
      auto normalize = [&](bool columns) {
        for (int64_t outer = 0; outer < streams; ++outer) {
          float sum = 0.0f;
          for (int64_t inner = 0; inner < streams; ++inner) {
            const int64_t index = columns ? inner * streams + outer : outer * streams + inner;
            sum += comb[static_cast<size_t>(comb_offset + index)];
          }
          sum += epsilon_;
          for (int64_t inner = 0; inner < streams; ++inner) {
            const int64_t index = columns ? inner * streams + outer : outer * streams + inner;
            comb[static_cast<size_t>(comb_offset + index)] /= sum;
          }
        }
      };
      normalize(true);
      for (int64_t iteration = 1; iteration < sinkhorn_iterations_; ++iteration) {
        normalize(false);
        normalize(true);
      }
    }

    std::vector<float> collapsed(static_cast<size_t>(rows * hidden_size), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t stream = 0; stream < streams; ++stream) {
        const float mix = pre[static_cast<size_t>(row * streams + stream)];
        for (int64_t d = 0; d < hidden_size; ++d) {
          collapsed[static_cast<size_t>(row * hidden_size + d)] +=
              mix * hidden_values[static_cast<size_t>((row * streams + stream) * hidden_size + d)];
        }
      }
    }

    TensorShapeVector post_shape{hidden->Shape()[0], hidden->Shape()[1], streams};
    TensorShapeVector comb_shape{hidden->Shape()[0], hidden->Shape()[1], streams, streams};
    TensorShapeVector collapsed_shape{hidden->Shape()[0], hidden->Shape()[1], hidden_size};
    WriteFloatVector<T>(*context->Output(0, post_shape), post);
    WriteFloatVector<T>(*context->Output(1, comb_shape), comb);
    WriteFloatVector<T>(*context->Output(2, collapsed_shape), collapsed);
    return Status::OK();
  }

 private:
  float epsilon_{};
  int64_t sinkhorn_iterations_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_HYPER_KERNEL(OP, T)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      OP, kMSDomain, 1, T, kCpuExecutionProvider,                   \
      (*KernelDefBuilder::Create())                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())   \
          .TypeConstraint("F", DataTypeImpl::GetTensorType<float>()), \
      deepseek_v4_attention_impl::OP<T>);

REGISTER_HYPER_KERNEL(HyperConnection, float)
REGISTER_HYPER_KERNEL(HyperConnection, MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime