// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/moe_cpu.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/float16.h"
#include "core/framework/allocator.h"
#include "core/platform/threadpool.h"

#include <algorithm>
#include <vector>
#include <numeric>

namespace onnxruntime {
namespace contrib {

template <typename T>
MoE<T>::MoE(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info), MoEBaseCPU(op_kernel_info) {
  if (activation_type_ == ActivationType::SwiGLU && !swiglu_interleaved_) {
    ORT_THROW("CPU MoE only supports interleaved SwiGLU format. Please set swiglu_fusion=1.");
  }
}

template <typename T>
Status MoE<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_experts_bias = context->Input<Tensor>(3);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias = context->Input<Tensor>(5);
  const Tensor* fc3_experts_weights = context->Input<Tensor>(6);
  const Tensor* fc3_experts_bias = context->Input<Tensor>(7);

  // FC3 not supported
  if (fc3_experts_weights != nullptr || fc3_experts_bias != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 is not implemented for CPU MoE.");
  }

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias, nullptr,
      fc2_experts_weights, fc2_experts_bias, nullptr,
      fc3_experts_weights, fc3_experts_bias, nullptr,
      1,
      activation_type_ == ActivationType::SwiGLU));

  Tensor* output = context->Output(0, input->Shape());

  return ComputeMoE(context, input, router_probs, fc1_experts_weights, fc1_experts_bias,
                    fc2_experts_weights, fc2_experts_bias, output);
}

template <typename T>
Status MoE<T>::ComputeMoE(const OpKernelContext* context,
                          const Tensor* input,
                          const Tensor* router_probs,
                          const Tensor* fc1_experts_weights,
                          const Tensor* fc1_experts_bias,
                          const Tensor* fc2_experts_weights,
                          const Tensor* fc2_experts_bias,
                          Tensor* output) const {
  const auto& input_shape = input->Shape();
  const auto& router_shape = router_probs->Shape();
  const auto& fc1_shape = fc1_experts_weights->Shape();
  const auto& fc2_shape = fc2_experts_weights->Shape();

  const int64_t num_tokens = input_shape.Size() / input_shape[input_shape.NumDimensions() - 1];
  const int64_t hidden_size = input_shape[input_shape.NumDimensions() - 1];
  const int64_t num_experts = router_shape[1];
  const int64_t inter_size = (fc2_shape[1] * fc2_shape[2]) / hidden_size;
  const bool is_swiglu = activation_type_ == ActivationType::SwiGLU;
  const int64_t fc1_output_size = is_swiglu ? (inter_size * 2) : inter_size;

  const T* input_data = input->Data<T>();
  const T* router_data = router_probs->Data<T>();
  const T* fc1_weights_data = fc1_experts_weights->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias ? fc1_experts_bias->Data<T>() : nullptr;
  const T* fc2_weights_data = fc2_experts_weights->Data<T>();
  const T* fc2_bias_data = fc2_experts_bias ? fc2_experts_bias->Data<T>() : nullptr;
  T* output_data = output->MutableData<T>();

  std::fill_n(output_data, output->Shape().Size(), T{});

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  IAllocatorUniquePtr<float> router_logits_float_buffer;
  const float* router_logits_float = nullptr;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    router_logits_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * num_experts));
    router_logits_float = router_logits_float_buffer.get();
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(router_data), const_cast<float*>(router_logits_float), static_cast<size_t>(num_tokens * num_experts));
  } else {
    router_logits_float = reinterpret_cast<const float*>(router_data);
  }

  auto route_expert_ptr = IAllocator::MakeUniquePtr<int>(allocator, static_cast<size_t>(num_tokens * k_));
  int* route_expert = route_expert_ptr.get();
  auto route_scale_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * k_));
  float* route_scale = route_scale_ptr.get();

  auto* tp = context->GetOperatorThreadPool();
  int num_routing_threads = (tp == nullptr || num_tokens < 4096) ? 1 : std::min(static_cast<int>(num_tokens), concurrency::ThreadPool::DegreeOfParallelism(tp));
  std::vector<std::vector<std::vector<int64_t>>> thread_local_expert_token_maps(num_routing_threads);
  for (auto& map : thread_local_expert_token_maps) {
    map.resize(static_cast<size_t>(num_experts));
  }

  concurrency::ThreadPool::TrySimpleParallelFor(tp, num_routing_threads, [&](std::ptrdiff_t thread_id) {
    auto work = concurrency::ThreadPool::PartitionWork(static_cast<int>(thread_id), num_routing_threads, static_cast<std::ptrdiff_t>(num_tokens));
    auto& local_expert_token_map = thread_local_expert_token_maps[thread_id];

    std::vector<std::pair<float, int64_t>> sorted_logits(static_cast<size_t>(num_experts));
    std::vector<float> top_k_exp(static_cast<size_t>(k_));

    for (int64_t i = work.start; i < work.end; ++i) {
      const float* logits = router_logits_float + i * num_experts;
      for (int64_t j = 0; j < num_experts; ++j) {
        sorted_logits[static_cast<size_t>(j)] = {logits[j], j};
      }
      std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + static_cast<std::ptrdiff_t>(k_), sorted_logits.end(), std::greater<>());

      if (normalize_routing_weights_) {
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < k_; ++j) {
          top_k_exp[static_cast<size_t>(j)] = std::exp(sorted_logits[static_cast<size_t>(j)].first);
          sum_exp += top_k_exp[static_cast<size_t>(j)];
        }

        float scale = (sum_exp == 0.0f) ? 0.0f : (1.0f / sum_exp);
        for (int64_t j = 0; j < k_; ++j) {
          int64_t expert_idx = sorted_logits[static_cast<size_t>(j)].second;
          int64_t route_idx = i * k_ + j;
          route_expert[route_idx] = static_cast<int>(expert_idx);
          route_scale[route_idx] = top_k_exp[static_cast<size_t>(j)] * scale;
          if (route_scale[route_idx] > 0.0f) {
            local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
          }
        }
      } else {
        for (int64_t j = 0; j < k_; ++j) {
          int64_t expert_idx = sorted_logits[static_cast<size_t>(j)].second;
          int64_t route_idx = i * k_ + j;
          route_expert[route_idx] = static_cast<int>(expert_idx);
          // logits points at the beginning of this token's expert values
          float val = logits[static_cast<size_t>(expert_idx)];
          route_scale[route_idx] = val;
          if (route_scale[route_idx] > 0.0f) {
            local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
          }
        }
      }
    }
  });

  std::vector<std::vector<int64_t>> expert_token_map(static_cast<size_t>(num_experts));
  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    size_t total_tokens_for_expert = 0;
    for (int t = 0; t < num_routing_threads; ++t) total_tokens_for_expert += thread_local_expert_token_maps[t][static_cast<size_t>(expert_idx)].size();
    expert_token_map[static_cast<size_t>(expert_idx)].reserve(total_tokens_for_expert);
  }
  for (int t = 0; t < num_routing_threads; ++t) {
    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      auto& local_tokens = thread_local_expert_token_maps[t][static_cast<size_t>(expert_idx)];
      if (!local_tokens.empty()) {
        expert_token_map[static_cast<size_t>(expert_idx)].insert(expert_token_map[static_cast<size_t>(expert_idx)].end(), local_tokens.begin(), local_tokens.end());
      }
    }
  }

  IAllocatorUniquePtr<float> input_float_buffer;
  const float* input_float;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    input_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * hidden_size));
    input_float = input_float_buffer.get();
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(input_data), const_cast<float*>(input_float), static_cast<size_t>(num_tokens * hidden_size));
  } else {
    input_float = reinterpret_cast<const float*>(input_data);
  }

  int num_expert_threads = (tp == nullptr) ? 1 : std::min(static_cast<int>(num_experts), concurrency::ThreadPool::DegreeOfParallelism(tp));
  if (num_expert_threads == 0) num_expert_threads = 1;
  const size_t output_buffer_size = static_cast<size_t>(output->Shape().Size());
  auto thread_local_outputs_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_expert_threads) * output_buffer_size);
  float* thread_local_outputs = thread_local_outputs_ptr.get();
  memset(thread_local_outputs, 0, static_cast<size_t>(num_expert_threads) * output_buffer_size * sizeof(float));

  concurrency::ThreadPool::TrySimpleParallelFor(tp, num_expert_threads, [&](std::ptrdiff_t thread_id_pd) {
    int thread_id = static_cast<int>(thread_id_pd);
    auto work = concurrency::ThreadPool::PartitionWork(thread_id, num_expert_threads, static_cast<std::ptrdiff_t>(num_experts));

    float* local_output = thread_local_outputs + static_cast<size_t>(thread_id) * output_buffer_size;

    for (int64_t expert_idx = work.start; expert_idx < work.end; ++expert_idx) {
      const auto& routes = expert_token_map[static_cast<size_t>(expert_idx)];
      if (routes.empty()) continue;

      const int64_t num_expert_tokens = static_cast<int64_t>(routes.size());

      std::vector<float> A1(static_cast<size_t>(num_expert_tokens * hidden_size));
      std::vector<float> batch_weights(static_cast<size_t>(num_expert_tokens));
      std::vector<int64_t> token_ids(static_cast<size_t>(num_expert_tokens));

      for (int64_t r = 0; r < num_expert_tokens; ++r) {
        int64_t route_idx = routes[static_cast<size_t>(r)];  // stored as i*k + j
        int64_t token = route_idx / k_;
        int64_t kth = route_idx % k_;
        token_ids[static_cast<size_t>(r)] = token;
        batch_weights[static_cast<size_t>(r)] = route_scale[route_idx];
        const float* src = input_float + token * hidden_size;
        float* dst = A1.data() + static_cast<size_t>(r) * static_cast<size_t>(hidden_size);
        std::copy(src, src + hidden_size, dst);
      }

      const T* fc1_expert_weights = fc1_weights_data + expert_idx * fc1_output_size * hidden_size;
      const T* fc1_expert_bias = fc1_bias_data ? fc1_bias_data + expert_idx * fc1_output_size : nullptr;
      const T* fc2_expert_weights = fc2_weights_data + expert_idx * hidden_size * inter_size;
      const T* fc2_expert_bias = fc2_bias_data ? fc2_bias_data + expert_idx * hidden_size : nullptr;

      std::vector<float> C2(static_cast<size_t>(num_expert_tokens * hidden_size));

      std::vector<T> A1_t(static_cast<size_t>(num_expert_tokens * hidden_size));
      for (size_t i = 0; i < A1_t.size(); ++i) A1_t[i] = static_cast<T>(A1[i]);

      std::vector<T> C2_t(static_cast<size_t>(num_expert_tokens * hidden_size));

      ORT_IGNORE_RETURN_VALUE(ProcessExpertBatch(A1_t.data(), token_ids.data(), batch_weights.data(),
                                                 num_expert_tokens, expert_idx,
                                                 fc1_expert_weights, fc1_expert_bias,
                                                 fc2_expert_weights, fc2_expert_bias,
                                                 C2_t.data(), hidden_size, inter_size));

      for (int64_t r = 0; r < num_expert_tokens; ++r) {
        int64_t token = token_ids[static_cast<size_t>(r)];
        const T* expert_output_t = C2_t.data() + static_cast<size_t>(r) * static_cast<size_t>(hidden_size);
        float w = batch_weights[static_cast<size_t>(r)];
        float* dest = local_output + static_cast<size_t>(token) * static_cast<size_t>(hidden_size);
        for (int64_t j = 0; j < hidden_size; ++j) dest[static_cast<size_t>(j)] += w * static_cast<float>(expert_output_t[static_cast<size_t>(j)]);
      }
    }
  });

  auto accumulate = [&](float* buffer) {
    memset(buffer, 0, output_buffer_size * sizeof(float));
    for (int i = 0; i < num_expert_threads; ++i) {
      const size_t thread_offset = static_cast<size_t>(i) * output_buffer_size;
      for (size_t j = 0; j < output_buffer_size; ++j) buffer[j] += thread_local_outputs[thread_offset + j];
    }
  };

  if constexpr (std::is_same_v<T, MLFloat16>) {
    auto final_output_float_ptr = IAllocator::MakeUniquePtr<float>(allocator, output_buffer_size);
    float* final_output_float = final_output_float_ptr.get();
    accumulate(final_output_float);

    MlasConvertFloatToHalfBuffer(final_output_float,
                                 reinterpret_cast<MLFloat16*>(output->MutableData<T>()),
                                 static_cast<size_t>(output_buffer_size));
  } else {
    auto final_output_float_ptr = IAllocator::MakeUniquePtr<float>(allocator, output_buffer_size);
    float* final_output_float = final_output_float_ptr.get();
    accumulate(final_output_float);

    float* out_ptr = reinterpret_cast<float*>(output->MutableData<T>());
    memcpy(out_ptr, final_output_float, output_buffer_size * sizeof(float));
  }
  return Status::OK();
}
template <typename T>
Status MoE<T>::ProcessExpertBatch(const T* input_tokens,
                                  const int64_t* token_expert_ids,
                                  const float* token_weights,
                                  int64_t batch_size,
                                  int64_t expert_id,
                                  const T* fc1_weights,
                                  const T* fc1_bias,
                                  const T* fc2_weights,
                                  const T* fc2_bias,
                                  T* output_buffer,
                                  int64_t hidden_size,
                                  int64_t inter_size) const {
  const bool is_swiglu = activation_type_ == ActivationType::SwiGLU;
  const int64_t fc1_output_size = is_swiglu ? (inter_size * 2) : inter_size;

  std::vector<T> fc1_output(batch_size * fc1_output_size);
  std::vector<T> activation_output(batch_size * inter_size);

  ORT_RETURN_IF_ERROR(ComputeGEMM(input_tokens, fc1_weights, fc1_output.data(),
                                  batch_size, hidden_size, fc1_output_size, true));

  if (fc1_bias) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      T* batch_output = fc1_output.data() + batch * fc1_output_size;
      for (int64_t i = 0; i < fc1_output_size; ++i) {
        batch_output[i] = static_cast<T>(static_cast<float>(batch_output[i]) +
                                         static_cast<float>(fc1_bias[i]));
      }
    }
  }

  if (is_swiglu) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      ApplySwiGLUVectorized(fc1_output.data() + batch * fc1_output_size,
                            activation_output.data() + batch * inter_size,
                            inter_size);
    }
  } else {
    ApplyActivationVectorized(fc1_output.data(), batch_size * fc1_output_size);
    activation_output.assign(fc1_output.begin(), fc1_output.end());
  }

  ORT_RETURN_IF_ERROR(ComputeGEMM(activation_output.data(), fc2_weights, output_buffer,
                                  batch_size, inter_size, hidden_size, true));

  if (fc2_bias) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      T* batch_output = output_buffer + batch * hidden_size;
      for (int64_t i = 0; i < hidden_size; ++i) {
        batch_output[i] = static_cast<T>(static_cast<float>(batch_output[i]) +
                                         static_cast<float>(fc2_bias[i]));
      }
    }
  }

  return Status::OK();
}

template <>
Status MoE<float>::ComputeGEMM(const float* A, const float* B, float* C,
                               int64_t M, int64_t K, int64_t N, bool transpose_B) const {
  MLAS_SGEMM_DATA_PARAMS params;
  params.A = A;
  params.lda = K;
  params.alpha = 1.0f;
  params.beta = 0.0f;
  params.C = C;
  params.ldc = N;
  params.B = B;

  if (transpose_B) {
    params.ldb = K;
    MlasGemm(CblasNoTrans, CblasTrans, M, N, K, params, nullptr);
  } else {
    params.ldb = N;
    MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, params, nullptr);
  }

  return Status::OK();
}

template <>
Status MoE<MLFloat16>::ComputeGEMM(const MLFloat16* A, const MLFloat16* B, MLFloat16* C,
                                   int64_t M, int64_t K, int64_t N, bool transpose_B) const {
  MLAS_HALF_GEMM_DATA_PARAMS params;
  params.A = A;
  params.lda = K;
  params.C = C;
  params.ldc = N;
  params.AIsfp32 = false;
  params.BIsfp32 = false;
  params.B = B;

  if (transpose_B) {
    params.ldb = K;
  } else {
    params.ldb = N;
  }

  MlasHalfGemmBatch(M, N, K, 1, &params, nullptr);
  return Status::OK();
}

template <typename T>
void MoE<T>::ApplyActivationVectorized(T* data, int64_t size) const {
  for (int64_t i = 0; i < size; ++i) {
    float val = static_cast<float>(data[i]);
    data[i] = static_cast<T>(ApplyActivation(val, activation_type_));
  }
}

template <typename T>
void MoE<T>::ApplySwiGLUVectorized(const T* input, T* output, int64_t size) const {
  for (int64_t i = 0; i < size; ++i) {
    float gate = static_cast<float>(input[2 * i]);
    float linear = static_cast<float>(input[2 * i + 1]);

    gate = std::min(gate, swiglu_limit_);
    linear = std::clamp(linear, -swiglu_limit_, swiglu_limit_);

    float sigmoid_out = 1.0f / (1.0f + std::exp(-activation_alpha_ * gate));
    float swish_out = gate * sigmoid_out;
    output[i] = static_cast<T>(swish_out * (linear + activation_beta_));
  }
}

template <>
void MoE<float>::ApplySwiGLUVectorized(const float* input, float* output, int64_t size) const {
  ApplySwiGLUActivation(input, output, size, true,
                        activation_alpha_, activation_beta_, swiglu_limit_);
}

template class MoE<float>;
template class MoE<MLFloat16>;

#define REGISTER_KERNEL_TYPED(type)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      MoE, kMSDomain, 1, type, kCpuExecutionProvider,                \
      (*KernelDefBuilder::Create())                                  \
          .MayInplace(0, 0)                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      MoE<type>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
