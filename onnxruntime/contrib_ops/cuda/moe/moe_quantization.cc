// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_quantization.h"
#include <type_traits>
#include "cutlass/numeric_types.h"
#include "core/common/safeint.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                               \
      QMoE,                                                                    \
      kMSDomain,                                                               \
      1,                                                                       \
      T,                                                                       \
      kCudaExecutionProvider,                                                  \
      (*KernelDefBuilder::Create())                                            \
          .MayInplace(0, 0)                                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())               \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())        \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),             \
      QMoE);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

QMoE::QMoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);

  this->block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", -1);

  using namespace onnxruntime::llm::kernels::cutlass_kernels;
  
  constexpr int kInputIndexFc3Weight = 8;
  has_fc3_ = op_kernel_info.GetInputCount() > kInputIndexFc3Weight && op_kernel_info.node().InputDefs()[kInputIndexFc3Weight]->Exists();

  int32_t input_type = op_kernel_info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  bool is_fp16 = input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;
  if (is_fp16) {
      if (expert_weight_bits_ == 4) {
          m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t, half>>(
              sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
      } else { // expert_weight_bits_ == 8
          m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, uint8_t, half>>(
              sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
      }
  } else { // BFloat16
      if (expert_weight_bits_ == 4) {
          m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16>>(
              sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
      } else { // expert_weight_bits_ == 8
          m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, uint8_t, __nv_bfloat16>>(
              sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
      }
  }
}

Status QMoE::ComputeInternal(OpKernelContext* context) const {
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

  // TODO: Add support for fc1_zeros and fc2_zeros
  const Tensor* fc1_zeros = nullptr;
  const Tensor* fc2_zeros = nullptr;

  // MoEQuantType quant_type = expert_weight_bits_ == 4 ? MoEQuantType::UINT4 : MoEQuantType::UINT8;

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, fc1_scales,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional,
      expert_weight_bits_ == 4 ? 2 : 1,
      activation_type_ == ActivationType::Swiglu));

  constexpr bool use_lora = false;
  constexpr bool use_deepseek_fp8_block_scale = false;
  constexpr bool min_latency_mode = false;
  bool use_awq = (fc1_zeros != nullptr);
  
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};

  size_t workspace_size = m_moe_runner->getWorkspaceSize(
      moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
      activation_type_, parallelism_config, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);
  auto work_space = GetScratchBuffer<void>(workspace_size, context->GetComputeStream());

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params;
  if (block_size_ > 0) {
      quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::GroupWise(
          block_size_,
          fc1_scales->DataRaw(),
          fc2_scales->DataRaw(),
          nullptr,
          nullptr,
          fc1_zeros ? fc1_zeros->DataRaw() : nullptr,
          fc2_zeros ? fc2_zeros->DataRaw() : nullptr);
  } else {
      // Per-column quantization
      quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::Int(
          fc1_scales->DataRaw(),
          fc2_scales->DataRaw());
  }

//   Tensor* output = context->Output(0, input->Shape());

//   // Note: The TRT-LLM kernel expects expert indices and scales, not raw probabilities.
//   // This assumes a TopK node has been placed before this operator in the ONNX graph.
//   const Tensor* expert_indices = router_probs; // Placeholder, should be output of TopK
//   const Tensor* expert_scales = router_probs; // Placeholder, should be output of TopK

//   m_moe_runner->runMoe(
//       input->DataRaw(),
//       nullptr,
//       expert_indices->Data<int>(),
//       expert_scales->Data<float>(),
//       fc1_experts_weights->DataRaw(),
//       fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr,
//       activation_type_,
//       fc2_experts_weights->DataRaw(),
//       fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr,
//       quant_params,
//       moe_params.num_rows,
//       moe_params.hidden_size,
//       moe_params.inter_size,
//       moe_params.num_experts,
//       k_,
//       reinterpret_cast<char*>(work_space.get()),
//       output->MutableDataRaw(),
//       nullptr,
//       {},
//       false,
//       false,
//       {},
//       false,
//       false,
//       {},
//       Stream(context));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
