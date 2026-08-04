// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "contrib_ops/cpu/bert/deepseek_v4_compression_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {
namespace {

constexpr int64_t kHyperConnectionMaxMult = 4;

void NormalizeAxis(float* matrix, int64_t order, float epsilon, bool by_row) {
  for (int64_t outer = 0; outer < order; ++outer) {
    float sum = 0.0f;
    for (int64_t inner = 0; inner < order; ++inner) {
      const int64_t index = by_row ? outer * order + inner : inner * order + outer;
      sum += matrix[index];
    }

    const float denominator = sum + epsilon;
    for (int64_t inner = 0; inner < order; ++inner) {
      const int64_t index = by_row ? outer * order + inner : inner * order + outer;
      matrix[index] /= denominator;
    }
  }
}

void NormalizeSinkhorn(float* matrix, int64_t order, int64_t iterations, float epsilon) {
  NormalizeAxis(matrix, order, epsilon, false);
  for (int64_t iteration = 1; iteration < iterations; ++iteration) {
    NormalizeAxis(matrix, order, epsilon, true);
    NormalizeAxis(matrix, order, epsilon, false);
  }
}

}  // namespace

class HyperConnectionMix final : public OpKernel {
 public:
  explicit HyperConnectionMix(const OpKernelInfo& info) : OpKernel(info) {
    sinkhorn_iterations_ = info.GetAttrOrDefault<int64_t>("sinkhorn_iterations", 1);
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-6f);
    hc_epsilon_ = info.GetAttrOrDefault<float>("hc_epsilon", 1e-6f);
    sinkhorn_epsilon_ = info.GetAttrOrDefault<float>("sinkhorn_epsilon", 1e-6f);
    post_alpha_ = info.GetAttrOrDefault<float>("post_alpha", 2.0f);
    ORT_ENFORCE(sinkhorn_iterations_ >= 1,
                "sinkhorn_iterations must be at least 1, got ", sinkhorn_iterations_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* x = context->Input<Tensor>(0);
    const Tensor* residual = context->Input<Tensor>(1);
    const Tensor* post_mix = context->Input<Tensor>(2);
    const Tensor* comb_mix = context->Input<Tensor>(3);
    const Tensor* fn = context->Input<Tensor>(4);
    const Tensor* scale = context->Input<Tensor>(5);
    const Tensor* base = context->Input<Tensor>(6);
    const Tensor* norm_weight = context->Input<Tensor>(7);
    ORT_RETURN_IF_NOT(x && residual && post_mix && comb_mix && fn && scale && base && norm_weight,
                      "HyperConnectionMix requires all inputs.");

    const auto& residual_dims = residual->Shape().GetDims();
    ORT_RETURN_IF_NOT(residual_dims.size() >= 3,
                      "residual is expected to have rank at least 3 (..., hc, dim), got ", residual_dims.size());

    const int64_t dim = residual_dims[residual_dims.size() - 1];
    const int64_t hc = residual_dims[residual_dims.size() - 2];
    ORT_RETURN_IF_NOT(hc >= 1 && hc <= kHyperConnectionMaxMult,
                      "The hyper-connection multiplicity must be in [1, ", kHyperConnectionMaxMult, "], got ", hc);
    ORT_RETURN_IF_NOT(dim >= 1, "The hidden dimension must be at least 1, got ", dim);

    const int64_t num_tokens = residual->Shape().Size() / (hc * dim);
    const int64_t mix_dim = (2 + hc) * hc;
    ORT_RETURN_IF_NOT(x->Shape().Size() == num_tokens * dim,
                      "x must hold ", num_tokens * dim, " elements to match residual, got ", x->Shape().Size());
    ORT_RETURN_IF_NOT(post_mix->Shape().Size() == num_tokens * hc &&
                          comb_mix->Shape().Size() == num_tokens * hc * hc,
                      "post_mix and comb_mix must be shaped (..., hc) and (..., hc, hc).");
    ORT_RETURN_IF_NOT(fn->Shape().Size() == hc * dim * mix_dim,
                      "fn must be shaped (hc * dim, ", mix_dim, ").");
    ORT_RETURN_IF_NOT(scale->Shape().Size() >= 3, "scale must hold at least 3 elements.");
    ORT_RETURN_IF_NOT(base->Shape().Size() == mix_dim, "base must hold ", mix_dim, " elements.");
    ORT_RETURN_IF_NOT(norm_weight->Shape().Size() == dim, "norm_weight must hold ", dim, " elements.");

    Tensor* residual_out = context->Output(0, residual->Shape());
    Tensor* post_mix_out = context->Output(1, post_mix->Shape());
    Tensor* comb_mix_out = context->Output(2, comb_mix->Shape());
    Tensor* layer_input = context->Output(3, x->Shape());

    const float* x_data = x->Data<float>();
    const float* residual_data = residual->Data<float>();
    const float* post_data = post_mix->Data<float>();
    const float* comb_data = comb_mix->Data<float>();
    const float* fn_data = fn->Data<float>();
    const float* scale_data = scale->Data<float>();
    const float* base_data = base->Data<float>();
    const float* norm_data = norm_weight->Data<float>();
    float* residual_output_data = residual_out->MutableData<float>();
    float* post_output_data = post_mix_out->MutableData<float>();
    float* comb_output_data = comb_mix_out->MutableData<float>();
    float* layer_input_data = layer_input->MutableData<float>();

    std::vector<float> projected(static_cast<size_t>(mix_dim));
    std::vector<float> pre(static_cast<size_t>(hc));
    std::vector<float> y(static_cast<size_t>(dim));
    for (int64_t token = 0; token < num_tokens; ++token) {
      std::fill(projected.begin(), projected.end(), 0.0f);
      float residual_square_sum = 0.0f;
      const int64_t x_offset = token * dim;
      const int64_t residual_offset = token * hc * dim;
      const int64_t post_offset = token * hc;
      const int64_t comb_offset = token * hc * hc;

      for (int64_t stream = 0; stream < hc; ++stream) {
        for (int64_t d = 0; d < dim; ++d) {
          float value = post_data[post_offset + stream] * x_data[x_offset + d];
          for (int64_t source = 0; source < hc; ++source) {
            value += comb_data[comb_offset + source * hc + stream] *
                     residual_data[residual_offset + source * dim + d];
          }
          residual_output_data[residual_offset + stream * dim + d] = value;
          residual_square_sum += value * value;

          const float* fn_row = fn_data + (stream * dim + d) * mix_dim;
          for (int64_t mix = 0; mix < mix_dim; ++mix) {
            projected[static_cast<size_t>(mix)] += value * fn_row[mix];
          }
        }
      }

      const float inverse_rms =
          1.0f / std::sqrt(residual_square_sum / static_cast<float>(hc * dim) + epsilon_);
      for (float& value : projected) {
        value *= inverse_rms;
      }

      for (int64_t stream = 0; stream < hc; ++stream) {
        const float pre_value = projected[static_cast<size_t>(stream)] * scale_data[0] + base_data[stream];
        pre[static_cast<size_t>(stream)] = 1.0f / (1.0f + std::exp(-pre_value)) + hc_epsilon_;

        const int64_t post_index = hc + stream;
        const float post_value = projected[static_cast<size_t>(post_index)] * scale_data[1] + base_data[post_index];
        post_output_data[post_offset + stream] =
            post_alpha_ / (1.0f + std::exp(-post_value));

        float maximum = -std::numeric_limits<float>::infinity();
        for (int64_t output_stream = 0; output_stream < hc; ++output_stream) {
          const int64_t mix_index = 2 * hc + stream * hc + output_stream;
          const float value = projected[static_cast<size_t>(mix_index)] * scale_data[2] + base_data[mix_index];
          maximum = std::max(maximum, value);
        }

        float denominator = 0.0f;
        for (int64_t output_stream = 0; output_stream < hc; ++output_stream) {
          const int64_t mix_index = 2 * hc + stream * hc + output_stream;
          const float value = projected[static_cast<size_t>(mix_index)] * scale_data[2] + base_data[mix_index];
          const size_t output_index = static_cast<size_t>(comb_offset + stream * hc + output_stream);
          comb_output_data[output_index] = std::exp(value - maximum);
          denominator += comb_output_data[output_index];
        }
        for (int64_t output_stream = 0; output_stream < hc; ++output_stream) {
          const size_t output_index = static_cast<size_t>(comb_offset + stream * hc + output_stream);
          comb_output_data[output_index] = comb_output_data[output_index] / denominator + hc_epsilon_;
        }
      }
      NormalizeSinkhorn(comb_output_data + comb_offset, hc, sinkhorn_iterations_, sinkhorn_epsilon_);

      float y_square_sum = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        float value = 0.0f;
        for (int64_t stream = 0; stream < hc; ++stream) {
          value += pre[static_cast<size_t>(stream)] *
                   residual_output_data[residual_offset + stream * dim + d];
        }
        y[static_cast<size_t>(d)] = value;
        y_square_sum += value * value;
      }

      const float y_inverse_rms = 1.0f / std::sqrt(y_square_sum / static_cast<float>(dim) + epsilon_);
      for (int64_t d = 0; d < dim; ++d) {
        layer_input_data[x_offset + d] = y[static_cast<size_t>(d)] * y_inverse_rms * norm_data[d];
      }
    }

    return Status::OK();
  }

 private:
  int64_t sinkhorn_iterations_{};
  float epsilon_{};
  float hc_epsilon_{};
  float sinkhorn_epsilon_{};
  float post_alpha_{};
};

}  // namespace deepseek_v4_attention_impl

ONNX_OPERATOR_TYPED_KERNEL_EX(
  HyperConnectionMix, kMSDomain, 1, float, kCpuExecutionProvider,
  (*KernelDefBuilder::Create())
    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
    .TypeConstraint("M", DataTypeImpl::GetTensorType<float>()),
  deepseek_v4_attention_impl::HyperConnectionMix);

}  // namespace contrib
}  // namespace onnxruntime
