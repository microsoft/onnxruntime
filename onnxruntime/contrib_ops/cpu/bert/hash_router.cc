// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/deepseek_v4_compression_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {

template <typename T>
class HashRouter final : public OpKernel {
 public:
  explicit HashRouter(const OpKernelInfo& info) : OpKernel(info) {
    score_function_ = info.GetAttrOrDefault<std::string>("score_function", "sigmoid");
    ORT_ENFORCE(score_function_ == "sigmoid" || score_function_ == "sqrtsoftplus",
                "HashRouter: score_function must be 'sigmoid' or 'sqrtsoftplus'.");
    routed_scaling_factor_ = info.GetAttrOrDefault<float>("routed_scaling_factor", 1.0f);
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-20f);
    ORT_ENFORCE(epsilon_ > 0.0f, "HashRouter: epsilon must be greater than zero.");
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* hidden = context->Input<Tensor>(0);
    const Tensor* input_ids = context->Input<Tensor>(1);
    const Tensor* gate_weight = context->Input<Tensor>(2);
    const Tensor* token_to_expert = context->Input<Tensor>(3);
    ORT_RETURN_IF_NOT(hidden && input_ids && gate_weight && token_to_expert, "HashRouter: all inputs are required.");
    ORT_RETURN_IF_NOT(hidden->Shape().NumDimensions() >= 1 && gate_weight->Shape().NumDimensions() == 2 &&
                          token_to_expert->Shape().NumDimensions() == 2,
                      "HashRouter: invalid input rank.");
    const int64_t hidden_size = hidden->Shape()[hidden->Shape().NumDimensions() - 1];
    const int64_t token_count = hidden->Shape().Size() / hidden_size;
    const int64_t num_experts = gate_weight->Shape()[0];
    const int64_t selected_count = token_to_expert->Shape()[1];
    const int64_t vocab_size = token_to_expert->Shape()[0];
    ORT_RETURN_IF_NOT(gate_weight->Shape()[1] == hidden_size && input_ids->Shape().Size() == token_count,
                      "HashRouter: input shapes are inconsistent.");
    for (size_t dimension = 0; dimension < input_ids->Shape().NumDimensions(); ++dimension) {
      ORT_RETURN_IF_NOT(input_ids->Shape()[dimension] == hidden->Shape()[dimension],
                        "input_ids shape must match the hidden_states prefix.");
    }
    ORT_RETURN_IF_NOT(input_ids->DataType() == token_to_expert->DataType(),
                      "input_ids and token_to_expert must have the same integer type.");

    const auto hidden_data = ToFloatVector<T>(*hidden);
    const auto gate_data = ToFloatVector<T>(*gate_weight);
    std::vector<float> logits(static_cast<size_t>(token_count * num_experts));
    for (int64_t token = 0; token < token_count; ++token) {
      for (int64_t expert = 0; expert < num_experts; ++expert) {
        float value = 0.0f;
        for (int64_t d = 0; d < hidden_size; ++d) {
          value += hidden_data[static_cast<size_t>(token * hidden_size + d)] *
                   gate_data[static_cast<size_t>(expert * hidden_size + d)];
        }
        logits[static_cast<size_t>(token * num_experts + expert)] = value;
      }
    }

    TensorShape logits_shape = hidden->Shape();
    logits_shape[logits_shape.NumDimensions() - 1] = num_experts;
    TensorShapeVector selected_dimensions(input_ids->Shape().GetDims().begin(),
                        input_ids->Shape().GetDims().end());
    selected_dimensions.push_back(selected_count);
    TensorShape selected_shape(selected_dimensions);
    WriteFloatVector<T>(*context->Output(0, logits_shape), logits);
    Tensor* routing_output = context->Output(1, selected_shape);
    Tensor* expert_output = context->Output(2, selected_shape);
    std::vector<float> routing(static_cast<size_t>(token_count * selected_count));

    auto compute = [&](const auto* ids, const auto* table, auto* output_ids) -> Status {
      for (int64_t token = 0; token < token_count; ++token) {
        const int64_t token_id = static_cast<int64_t>(ids[token]);
        ORT_RETURN_IF(token_id < 0 || token_id >= vocab_size, "HashRouter: input id is outside token_to_expert.");
        float denominator = 0.0f;
        for (int64_t index = 0; index < selected_count; ++index) {
          const int64_t expert = static_cast<int64_t>(table[token_id * selected_count + index]);
          ORT_RETURN_IF(expert < 0 || expert >= num_experts, "HashRouter: frozen expert id is out of range.");
          output_ids[token * selected_count + index] = static_cast<std::remove_pointer_t<decltype(output_ids)>>(expert);
          const float logit = logits[static_cast<size_t>(token * num_experts + expert)];
          const float score = score_function_ == "sigmoid"
                                  ? 1.0f / (1.0f + std::exp(-logit))
                                  : std::sqrt(std::max(0.0f, std::log1p(std::exp(-std::abs(logit))) +
                                                                std::max(logit, 0.0f)));
          routing[static_cast<size_t>(token * selected_count + index)] = score;
          denominator += score;
        }
        denominator = std::max(denominator, epsilon_);
        for (int64_t index = 0; index < selected_count; ++index) {
          routing[static_cast<size_t>(token * selected_count + index)] *= routed_scaling_factor_ / denominator;
        }
      }
      return Status::OK();
    };
    if (input_ids->IsDataType<int64_t>()) {
      ORT_RETURN_IF_ERROR(compute(input_ids->Data<int64_t>(), token_to_expert->Data<int64_t>(),
                                  expert_output->MutableData<int64_t>()));
    } else {
      ORT_RETURN_IF_NOT(input_ids->IsDataType<int32_t>(), "HashRouter: ids must be int32 or int64.");
      ORT_RETURN_IF_ERROR(compute(input_ids->Data<int32_t>(), token_to_expert->Data<int32_t>(),
                                  expert_output->MutableData<int32_t>()));
    }
    WriteFloatVector<T>(*routing_output, routing);
    return Status::OK();
  }

 private:
  std::string score_function_;
  float routed_scaling_factor_{};
  float epsilon_{};
};



}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      HashRouter, kMSDomain, 1, T, kCpuExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", {DataTypeImpl::GetTensorType<int32_t>(), \
                                  DataTypeImpl::GetTensorType<int64_t>()}), \
      deepseek_v4_attention_impl::HashRouter<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
