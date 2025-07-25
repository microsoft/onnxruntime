// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/moe_quantization_cpu.h"

#include <algorithm>
#include <vector>
#include <cmath>

#include "core/common/safeint.h"
#include "core/framework/float16.h"
#include "core/framework/int4.h"

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cpu {

QMoE::QMoE(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info), MoEBaseCPU(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
}

Status QMoE::QuantizedMoEImpl_UInt4x2(OpKernelContext* context,
                                      MoEParameters& moe_params,
                                      const Tensor* input,
                                      const Tensor* router_probs,
                                      const Tensor* fc1_experts_weights,
                                      const Tensor* fc1_experts_bias_optional,
                                      const Tensor* fc2_experts_weights,
                                      const Tensor* fc2_experts_bias_optional,
                                      const Tensor* fc3_experts_weights_optional,
                                      const Tensor* fc3_experts_bias_optional,
                                      const Tensor* fc1_scales,
                                      const Tensor* fc2_scales,
                                      const Tensor* fc3_scales_optional) const {
  // FC3 (gating) check - throw error if present
  if (fc3_experts_weights_optional != nullptr) {
    ORT_THROW("FC3 gating is not yet implemented for CPU quantized MoE. Please use the CUDA version for gated experts or disable FC3 gating.");
  }

  // Create output tensor
  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();

  // Simple implementation: just return zeros for now
  std::fill(output_data, output_data + moe_params.num_rows * moe_params.hidden_size, MLFloat16(0.0f));

  return Status::OK();
}

Status QMoE::QuantizedMoEImpl_UInt8(OpKernelContext* context,
                                    MoEParameters& moe_params,
                                    const Tensor* input,
                                    const Tensor* router_probs,
                                    const Tensor* fc1_experts_weights,
                                    const Tensor* fc1_experts_bias_optional,
                                    const Tensor* fc2_experts_weights,
                                    const Tensor* fc2_experts_bias_optional,
                                    const Tensor* fc3_experts_weights_optional,
                                    const Tensor* fc3_experts_bias_optional,
                                    const Tensor* fc1_scales,
                                    const Tensor* fc2_scales,
                                    const Tensor* fc3_scales_optional) const {
  // FC3 (gating) check - throw error if present
  if (fc3_experts_weights_optional != nullptr) {
    ORT_THROW("FC3 gating is not yet implemented for CPU quantized MoE. Please use the CUDA version for gated experts or disable FC3 gating.");
  }

  // Create output tensor
  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();

  // Simple implementation: just return zeros for now
  std::fill(output_data, output_data + moe_params.num_rows * moe_params.hidden_size, MLFloat16(0.0f));

  return Status::OK();
}

Status QMoE::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_scales = context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(5);
  const Tensor* fc2_scales = context->Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context->Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(10);

  MoEQuantType quant_type = expert_weight_bits_ == 4 ? MoEQuantType::UINT4 : MoEQuantType::UINT8;
  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(CheckInputs(moe_params, quant_type, input, router_probs, fc1_experts_weights,
                                  fc1_experts_bias_optional, fc2_experts_weights, fc2_experts_bias_optional,
                                  fc3_experts_weights_optional, fc3_experts_bias_optional));
  ORT_RETURN_IF_ERROR(CheckInputScales(fc1_scales, fc2_scales, fc3_scales_optional, moe_params.num_experts,
                                       moe_params.hidden_size, moe_params.inter_size));

  if (quant_type == MoEQuantType::UINT4) {
    return QuantizedMoEImpl_UInt4x2(context, moe_params, input, router_probs,
                                    fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                    fc2_experts_bias_optional, fc3_experts_weights_optional,
                                    fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
  } else {
    return QuantizedMoEImpl_UInt8(context, moe_params, input, router_probs,
                                  fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                  fc2_experts_bias_optional, fc3_experts_weights_optional,
                                  fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
  }
}

}  // namespace cpu
}  // namespace contrib
}  // namespace onnxruntime
