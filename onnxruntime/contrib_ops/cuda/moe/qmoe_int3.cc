// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_quantization.h"
#include "contrib_ops/cuda/moe/qmoe_int3.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/common/narrow.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace onnxruntime::contrib::cuda {

Status QMoE::PrePackInt3(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed) {
  if (input_idx != 2 && input_idx != 3 && input_idx != 5 && input_idx != 6 && input_idx != 11 && input_idx != 12)
    return Status::OK();
  auto& entry = int3_inputs_[input_idx];
  ORT_RETURN_IF_NOT(!entry.data, "INT3 QMoE initializer was already packed.");
  const bool weight = input_idx == 2 || input_idx == 5;
  const bool scale = input_idx == 3 || input_idx == 6;
  ORT_RETURN_IF_NOT(scale ? (is_fp16_ ? tensor.IsDataType<MLFloat16>() : tensor.IsDataType<BFloat16>())
                          : tensor.IsDataType<uint8_t>(),
                    "INT3 QMoE initializer type mismatch.");
  const auto& shape = tensor.Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 3 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0,
                    "INT3 QMoE initializer must have positive rank-three shape.");
  entry.shape = shape;
  entry.dtype = tensor.DataType();
  entry.bits = narrow<int>((input_idx == 2 || input_idx == 3 || input_idx == 11)
                               ? fc1_expert_weight_bits_
                               : fc2_expert_weight_bits_);
  entry.block_size = block_size_;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&entry.device));
  entry.device_uuid = GetDeviceProp().uuid;
  struct PrepackStream {
    cudaStream_t handle = nullptr;
    ~PrepackStream() {
      if (handle) {
        cudaStreamSynchronize(handle);
        cudaStreamDestroy(handle);
      }
    }
  } prepack_stream;
  CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&prepack_stream.handle, cudaStreamNonBlocking));
  const auto stream = prepack_stream.handle;
  bool copied = false;
  PrePackCopyToGpu(tensor, stream, alloc, entry.data, copied);
  if (weight) {
    const int64_t logical_k = SafeInt<int64_t>(shape[2]) * 8 / entry.bits;
    ORT_RETURN_IF_NOT(logical_k > 0 && logical_k % block_size_ == 0 &&
                          moe_helper::PackedByteCountWithPadding(logical_k, entry.bits) == shape[2],
                      "INT3 QMoE packed weight K must be block aligned.");
    auto packed = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
    CUDA_RETURN_IF_ERROR(PackInt3MoeWeights(static_cast<const uint8_t*>(entry.data.get()),
                                            static_cast<uint8_t*>(packed.get()), SafeInt<size_t>(shape[0]) * shape[1],
                                            narrow<int>(logical_k), entry.bits, stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    entry.data = std::move(packed);
    entry.layout_version = 1;
  }
  auto invalid = IAllocator::MakeUniquePtr<int>(alloc, 1, true);
  for (int projection : {0, 1}) {
    auto& weights = int3_inputs_[projection ? 5 : 2];
    auto& scales = int3_inputs_[projection ? 6 : 3];
    auto& zeros = int3_inputs_[projection ? 12 : 11];
    Int3Projection check{};
    check.bits = narrow<int>(projection ? fc2_expert_weight_bits_ : fc1_expert_weight_bits_);
    if (scales.data && !scales.validated) {
      check.scales = scales.data.get();
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(invalid.get(), 0, sizeof(int), stream));
      CUDA_RETURN_IF_ERROR(ValidateInt3MoeContent(check, SafeInt<size_t>(scales.shape[0]) * scales.shape[1],
                                                  narrow<int>(SafeInt<int64_t>(scales.shape[2]) * block_size_), narrow<int>(block_size_), !is_fp16_,
                                                  invalid.get(), stream));
      int result = 0;
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(&result, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
      ORT_RETURN_IF_NOT(!result, "INT3 QMoE requires finite nonnegative scales and INT3 zero points equal to 4 with zero padding.");
      scales.validated = true;
    }
    if (zeros.data && !zeros.validated && (weights.data || scales.data)) {
      const int64_t logical_k = weights.data ? SafeInt<int64_t>(weights.shape[2]) * 8 / check.bits
                                             : SafeInt<int64_t>(scales.shape[2]) * block_size_;
      ORT_RETURN_IF_NOT(zeros.shape[2] == moe_helper::PackedByteCountWithPadding(logical_k / block_size_, check.bits),
                        "INT3 QMoE zero-point shape mismatch.");
      check.scales = nullptr;
      check.zeros = static_cast<const uint8_t*>(zeros.data.get());
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(invalid.get(), 0, sizeof(int), stream));
      CUDA_RETURN_IF_ERROR(ValidateInt3MoeContent(check, SafeInt<size_t>(zeros.shape[0]) * zeros.shape[1],
                                                  narrow<int>(logical_k), narrow<int>(block_size_), !is_fp16_, invalid.get(), stream));
      int result = 0;
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(&result, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
      ORT_RETURN_IF_NOT(!result, "INT3 QMoE requires finite nonnegative scales and INT3 zero points equal to 4 with zero padding.");
      zeros.validated = true;
    }
  }
  is_packed = true;
  return Status::OK();
}

Status QMoE::ComputeInt3(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto* router = context->Input<Tensor>(1);
  const auto* first_weights = int3_inputs_[2].data ? nullptr : context->Input<Tensor>(2);
  const auto* first_scales = int3_inputs_[3].data ? nullptr : context->Input<Tensor>(3);
  const auto* first_bias = context->Input<Tensor>(4);
  const auto* second_weights = int3_inputs_[5].data ? nullptr : context->Input<Tensor>(5);
  const auto* second_scales = int3_inputs_[6].data ? nullptr : context->Input<Tensor>(6);
  const auto* second_bias = context->Input<Tensor>(7);
  const auto* first_zeros = int3_inputs_[11].data ? nullptr : context->Input<Tensor>(11);
  const auto* second_zeros = int3_inputs_[12].data ? nullptr : context->Input<Tensor>(12);
  const auto* router_weights = context->Input<Tensor>(14);
  for (int index : {8, 9, 10, 13, 15, 16, 17, 18, 19, 20}) {
    ORT_RETURN_IF_NOT(!context->Input<Tensor>(index), "INT3 QMoE does not accept input ", index, ".");
  }
  ORT_RETURN_IF_NOT(input && router && (first_weights || int3_inputs_[2].data) &&
                        (first_scales || int3_inputs_[3].data) && (second_weights || int3_inputs_[5].data) &&
                        (second_scales || int3_inputs_[6].data),
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
  const auto shape_of = [&](int index, const Tensor* tensor) -> const TensorShape& {
    return int3_inputs_[index].data ? int3_inputs_[index].shape : tensor->Shape();
  };
  const auto data_of = [&](int index, const Tensor* tensor) -> const void* {
    return int3_inputs_[index].data ? int3_inputs_[index].data.get() : (tensor ? tensor->DataRaw() : nullptr);
  };
  int device = -1;
  const auto& properties = GetDeviceProp();
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&device));
  for (int index : {2, 3, 5, 6, 11, 12}) {
    const auto& entry = int3_inputs_[index];
    if (!entry.data) continue;
    ORT_RETURN_IF_NOT(entry.device == device && !std::memcmp(&entry.device_uuid, &properties.uuid, sizeof(properties.uuid)) &&
                          entry.block_size == block_size_ && entry.bits == ((index == 2 || index == 3 || index == 11) ? fc1_expert_weight_bits_ : fc2_expert_weight_bits_) &&
                          entry.layout_version == ((index == 2 || index == 5) ? 1 : 0),
                      "INT3 QMoE private cache identity mismatch.");
    ORT_RETURN_IF_NOT((index != 3 && index != 6) || entry.dtype == input->DataType(),
                      "INT3 QMoE floating input types must match activations.");
  }
  const auto& first_shape = shape_of(2, first_weights);
  const auto& second_shape = shape_of(5, second_weights);
  ORT_RETURN_IF_NOT(first_shape.NumDimensions() == 3 && second_shape.NumDimensions() == 3,
                    "INT3 QMoE weights must have rank 3.");
  const int64_t first_rows = first_shape[1];
  ORT_RETURN_IF_NOT(first_rows > 0 && first_rows % 2 == 0,
                    "INT3 QMoE FC1 must contain interleaved gate/up rows.");
  const int64_t intermediate = first_rows / 2;
  ORT_RETURN_IF_NOT(intermediate % block_size_ == 0, "INT3 QMoE intermediate size must be block aligned.");
  const int64_t experts = first_shape[0];
  ORT_RETURN_IF_NOT(experts > 0 && k_ > 0 && k_ <= experts, "INT3 QMoE requires 0 < k <= num_experts.");
  const int64_t rows = input_shape.Size() / hidden;
  ORT_RETURN_IF_NOT(router->Shape() == TensorShape({rows, experts}), "INT3 QMoE router shape mismatch.");
  ORT_RETURN_IF_NOT(!router_weights || router_weights->Shape() == router->Shape(),
                    "INT3 QMoE router_weights shape mismatch.");
  auto check_projection = [&](int weight_index, int scale_index, int zero_index,
                              const Tensor* weights, const Tensor* scales, const Tensor* zeros,
                              const Tensor* bias, int64_t outputs, int64_t features, int64_t bits) -> Status {
    ORT_RETURN_IF_NOT(shape_of(weight_index, weights) == TensorShape({experts, outputs, moe_helper::PackedByteCountWithPadding(features, bits)}),
                      "INT3 QMoE raw projection shape mismatch.");
    ORT_RETURN_IF_NOT(shape_of(scale_index, scales) == TensorShape({experts, outputs, features / block_size_}),
                      "INT3 QMoE scale shape mismatch.");
    ORT_RETURN_IF_NOT((!zeros && !int3_inputs_[zero_index].data) || shape_of(zero_index, zeros) == TensorShape({experts, outputs,
                                                                                                                moe_helper::PackedByteCountWithPadding(features / block_size_, bits)}),
                      "INT3 QMoE zero-point shape mismatch.");
    ORT_RETURN_IF_NOT(!bias || bias->Shape() == TensorShape({experts, outputs}), "INT3 QMoE bias shape mismatch.");
    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(check_projection(2, 3, 11, first_weights, first_scales, first_zeros, first_bias,
                                       first_rows, hidden, fc1_expert_weight_bits_));
  ORT_RETURN_IF_ERROR(check_projection(5, 6, 12, second_weights, second_scales, second_zeros, second_bias,
                                       hidden, intermediate, fc2_expert_weight_bits_));
  Int3MoeParams params{};
  params.fc1 = {static_cast<const uint8_t*>(data_of(2, first_weights)), data_of(3, first_scales),
                static_cast<const uint8_t*>(data_of(11, first_zeros)),
                first_bias ? first_bias->DataRaw() : nullptr, narrow<int>(fc1_expert_weight_bits_), int3_inputs_[2].layout_version};
  params.fc2 = {static_cast<const uint8_t*>(data_of(5, second_weights)), data_of(6, second_scales),
                static_cast<const uint8_t*>(data_of(12, second_zeros)),
                second_bias ? second_bias->DataRaw() : nullptr, narrow<int>(fc2_expert_weight_bits_), int3_inputs_[5].layout_version};
  params.hidden = narrow<int>(hidden);
  params.intermediate = narrow<int>(intermediate);
  params.top_k = narrow<int>(k_);
  params.num_experts = narrow<int>(experts);
  params.block_size = narrow<int>(block_size_);
  params.alpha = activation_alpha_;
  params.beta = activation_beta_;
  params.limit = swiglu_limit_;
  params.is_bfloat16 = !input->IsDataType<MLFloat16>();
  auto* compute_stream = GetComputeStream(context);
  const auto stream = compute_stream ? static_cast<cudaStream_t>(compute_stream->GetHandle()) : nullptr;
  auto first_check = params.fc1;
  auto second_check = params.fc2;
  if (int3_inputs_[3].validated) first_check.scales = nullptr;
  if (int3_inputs_[6].validated) second_check.scales = nullptr;
  if (int3_inputs_[11].validated) first_check.zeros = nullptr;
  if (int3_inputs_[12].validated) second_check.zeros = nullptr;
  if (first_check.scales || second_check.scales ||
      (first_check.zeros && first_check.bits == 3) || (second_check.zeros && second_check.bits == 3)) {
    auto invalid = GetScratchBuffer<int>(1, compute_stream);
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(invalid.get(), 0, sizeof(int), stream));
    CUDA_RETURN_IF_ERROR(ValidateInt3MoeContent(first_check, SafeInt<size_t>(experts) * first_rows,
                                                params.hidden, params.block_size, params.is_bfloat16, invalid.get(), stream));
    CUDA_RETURN_IF_ERROR(ValidateInt3MoeContent(second_check, SafeInt<size_t>(experts) * hidden,
                                                params.intermediate, params.block_size, params.is_bfloat16, invalid.get(), stream));
    int invalid_host = 0;
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(&invalid_host, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    ORT_RETURN_IF_NOT(invalid_host == 0, "INT3 QMoE requires finite nonnegative scales and INT3 zero points equal to 4 with zero padding.");
  }
  auto* output = context->Output(0, input_shape);
  if (rows == 0) return Status::OK();
  const size_t element_size = input->DataType()->Size();
  const size_t per_row_bytes = SafeInt<size_t>(k_) * (SafeInt<size_t>(intermediate) * element_size + 3 * sizeof(int));
  const size_t budget = std::min<size_t>(int_dequant_max_scratch_bytes_, size_t{64} << 20);
  ORT_RETURN_IF_NOT(per_row_bytes <= budget, "INT3 QMoE scratch budget cannot accommodate one token.");
  size_t tile_rows = std::min<size_t>(rows, budget / per_row_bytes);
  if (row_tile_size_ > 0) tile_rows = std::min<size_t>(tile_rows, row_tile_size_);
  tile_rows = std::min<size_t>(tile_rows, std::numeric_limits<int>::max() /
                                              std::max({hidden, static_cast<int64_t>(SafeInt<int64_t>(k_) * intermediate), experts}));
  ORT_RETURN_IF_NOT(tile_rows > 0, "INT3 QMoE dimensions exceed launch limits.");
  const size_t routes = SafeInt<size_t>(tile_rows) * k_;
  auto selected = GetScratchBuffer<int>(routes, compute_stream);
  auto scales = GetScratchBuffer<float>(routes, compute_stream);
  auto source_rows = GetScratchBuffer<int>(routes, compute_stream);
  auto activation = GetScratchBuffer<void>(SafeInt<size_t>(routes) * intermediate * element_size, compute_stream);
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
