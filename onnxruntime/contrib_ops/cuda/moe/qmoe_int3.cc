#include "contrib_ops/cuda/moe/moe_quantization.h"
#include "contrib_ops/cuda/moe/qmoe_int3.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/common/narrow.h"

#include <algorithm>
#include <limits>

namespace onnxruntime::contrib::cuda {

Status QMoE::ComputeInt3(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto* router = context->Input<Tensor>(1);
  const auto* first_weights = context->Input<Tensor>(2);
  const auto* first_scales = context->Input<Tensor>(3);
  const auto* first_bias = context->Input<Tensor>(4);
  const auto* second_weights = context->Input<Tensor>(5);
  const auto* second_scales = context->Input<Tensor>(6);
  const auto* second_bias = context->Input<Tensor>(7);
  const auto* first_zeros = context->Input<Tensor>(11);
  const auto* second_zeros = context->Input<Tensor>(12);
  const auto* router_weights = context->Input<Tensor>(14);
  for (int index : {8, 9, 10, 13, 15, 16, 17, 18, 19, 20}) {
    ORT_RETURN_IF_NOT(!context->Input<Tensor>(index), "INT3 QMoE does not accept input ", index, ".");
  }
  ORT_RETURN_IF_NOT(input && router && first_weights && first_scales && second_weights && second_scales,
                    "INT3 QMoE requires inputs 0, 1, 2, 3, 5, and 6.");
  ORT_RETURN_IF_NOT(input->IsDataType<MLFloat16>() || input->IsDataType<BFloat16>(),
                    "INT3 QMoE requires FP16/BF16 activations.");
  for (const auto* tensor : {router, first_scales, first_bias, second_scales, second_bias, router_weights}) {
    ORT_RETURN_IF_NOT(!tensor || tensor->DataType() == input->DataType(),
                      "INT3 QMoE floating input types must match activations.");
  }
  for (const auto* tensor : {first_weights, second_weights, first_zeros, second_zeros}) {
    ORT_RETURN_IF_NOT(!tensor || tensor->IsDataType<uint8_t>(), "INT3 QMoE packed tensors must be uint8.");
  }
  const auto& input_shape = input->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2 || input_shape.NumDimensions() == 3,
                    "INT3 QMoE input must have rank 2 or 3.");
  const int64_t hidden = input_shape.GetDims().back();
  ORT_RETURN_IF_NOT(hidden > 0 && hidden % block_size_ == 0, "INT3 QMoE hidden size must be positive and block aligned.");
  ORT_RETURN_IF_NOT(first_weights->Shape().NumDimensions() == 3 && second_weights->Shape().NumDimensions() == 3,
                    "INT3 QMoE weights must have rank 3.");
  const int64_t first_rows = first_weights->Shape()[1];
  ORT_RETURN_IF_NOT(first_rows > 0 && first_rows % 2 == 0,
                    "INT3 QMoE FC1 must contain interleaved gate/up rows.");
  const int64_t intermediate = first_rows / 2;
  ORT_RETURN_IF_NOT(intermediate % block_size_ == 0, "INT3 QMoE intermediate size must be block aligned.");
  const int64_t experts = first_weights->Shape()[0];
  ORT_RETURN_IF_NOT(experts > 0 && k_ > 0 && k_ <= experts, "INT3 QMoE requires 0 < k <= num_experts.");
  const int64_t rows = input_shape.Size() / hidden;
  ORT_RETURN_IF_NOT(router->Shape() == TensorShape({rows, experts}), "INT3 QMoE router shape mismatch.");
  ORT_RETURN_IF_NOT(!router_weights || router_weights->Shape() == router->Shape(),
                    "INT3 QMoE router_weights shape mismatch.");
  auto check_projection = [&](const Tensor* weights, const Tensor* scales, const Tensor* zeros,
                              const Tensor* bias, int64_t outputs, int64_t features, int64_t bits) -> Status {
    ORT_RETURN_IF_NOT(weights->Shape() == TensorShape({experts, outputs, moe_helper::PackedByteCountWithPadding(features, bits)}),
                      "INT3 QMoE raw projection shape mismatch.");
    ORT_RETURN_IF_NOT(scales->Shape() == TensorShape({experts, outputs, features / block_size_}),
                      "INT3 QMoE scale shape mismatch.");
    ORT_RETURN_IF_NOT(!zeros || zeros->Shape() == TensorShape({experts, outputs,
                                                               moe_helper::PackedByteCountWithPadding(features / block_size_, bits)}),
                      "INT3 QMoE zero-point shape mismatch.");
    ORT_RETURN_IF_NOT(!bias || bias->Shape() == TensorShape({experts, outputs}), "INT3 QMoE bias shape mismatch.");
    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(check_projection(first_weights, first_scales, first_zeros, first_bias,
                                       first_rows, hidden, fc1_expert_weight_bits_));
  ORT_RETURN_IF_ERROR(check_projection(second_weights, second_scales, second_zeros, second_bias,
                                       hidden, intermediate, fc2_expert_weight_bits_));
  Int3MoeParams params{};
  params.fc1 = {first_weights->Data<uint8_t>(), first_scales->DataRaw(),
                first_zeros ? first_zeros->Data<uint8_t>() : nullptr,
                first_bias ? first_bias->DataRaw() : nullptr, narrow<int>(fc1_expert_weight_bits_)};
  params.fc2 = {second_weights->Data<uint8_t>(), second_scales->DataRaw(),
                second_zeros ? second_zeros->Data<uint8_t>() : nullptr,
                second_bias ? second_bias->DataRaw() : nullptr, narrow<int>(fc2_expert_weight_bits_)};
  params.hidden = narrow<int>(hidden);
  params.intermediate = narrow<int>(intermediate);
  params.top_k = narrow<int>(k_);
  params.num_experts = narrow<int>(experts);
  params.block_size = narrow<int>(block_size_);
  params.alpha = activation_alpha_;
  params.beta = activation_beta_;
  params.limit = swiglu_limit_;
  params.is_bfloat16 = !input->IsDataType<MLFloat16>();
  const auto stream = Stream(context);
  auto invalid = GetScratchBuffer<int>(1, GetComputeStream(context));
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(invalid.get(), 0, sizeof(int), stream));
  CUDA_RETURN_IF_ERROR(ValidateInt3MoeContent(params.fc1, SafeInt<size_t>(experts) * first_rows,
                                              params.hidden, params.block_size, params.is_bfloat16, invalid.get(), stream));
  CUDA_RETURN_IF_ERROR(ValidateInt3MoeContent(params.fc2, SafeInt<size_t>(experts) * hidden,
                                              params.intermediate, params.block_size, params.is_bfloat16, invalid.get(), stream));
  int invalid_host = 0;
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(&invalid_host, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
  ORT_RETURN_IF_NOT(invalid_host == 0, "INT3 QMoE requires finite nonnegative scales and INT3 zero points equal to 4 with zero padding.");
  auto* output = context->Output(0, input_shape);
  if (rows == 0) return Status::OK();
  const size_t element_size = input->DataType()->Size();
  const size_t per_row_bytes = SafeInt<size_t>(k_) * (SafeInt<size_t>(intermediate) * element_size + 3 * sizeof(int));
  const size_t budget = std::min<size_t>(int_dequant_max_scratch_bytes_, size_t{64} << 20);
  ORT_RETURN_IF_NOT(per_row_bytes <= budget, "INT3 QMoE scratch budget cannot accommodate one token.");
  size_t tile_rows = std::min<size_t>(rows, budget / per_row_bytes);
  if (row_tile_size_ > 0) tile_rows = std::min<size_t>(tile_rows, row_tile_size_);
  tile_rows = std::min<size_t>(tile_rows, std::numeric_limits<int>::max() / std::max<int64_t>(hidden, SafeInt<int64_t>(k_) * intermediate));
  ORT_RETURN_IF_NOT(tile_rows > 0, "INT3 QMoE dimensions exceed launch limits.");
  const size_t routes = SafeInt<size_t>(tile_rows) * k_;
  auto selected = GetScratchBuffer<int>(routes, GetComputeStream(context));
  auto scales = GetScratchBuffer<float>(routes, GetComputeStream(context));
  auto source_rows = GetScratchBuffer<int>(routes, GetComputeStream(context));
  auto activation = GetScratchBuffer<void>(SafeInt<size_t>(routes) * intermediate * element_size, GetComputeStream(context));
  params.activation = activation.get();
  params.experts = selected.get();
  params.routing_scales = scales.get();
  for (int64_t offset = 0; offset < rows; offset += tile_rows) {
    params.rows = narrow<int>(std::min<size_t>(tile_rows, rows - offset));
    const auto route = [&](const auto* logits) {
      if (use_sparse_mixer_) {
        LaunchSparseMixerTop2(logits, scales.get(), selected.get(), source_rows.get(),
                              params.rows, params.num_experts, stream);
      } else {
        LaunchSoftmaxTopK(logits, scales.get(), selected.get(), params.rows, params.num_experts,
                          params.top_k, normalize_routing_weights_, stream);
      }
    };
    if (params.is_bfloat16)
      route(reinterpret_cast<const __nv_bfloat16*>(router->DataRaw()) + offset * experts);
    else
      route(reinterpret_cast<const half*>(router->DataRaw()) + offset * experts);
    params.input = static_cast<const char*>(input->DataRaw()) + offset * hidden * element_size;
    params.output = static_cast<char*>(output->MutableDataRaw()) + offset * hidden * element_size;
    params.router_weights = router_weights
                                ? static_cast<const char*>(router_weights->DataRaw()) + offset * experts * element_size
                                : nullptr;
    CUDA_RETURN_IF_ERROR(LaunchInt3Moe(params, stream));
  }
  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda