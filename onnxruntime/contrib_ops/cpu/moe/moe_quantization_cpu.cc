// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/moe_quantization_cpu.h"

#include "core/framework/allocator.h"
#include "core/framework/float16.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/common/safeint.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/util/math.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include "contrib_ops/cpu/moe/moe_helper.h"

#include <atomic>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory_resource>

static constexpr size_t kMemoryAlignment = 32;
static constexpr size_t kMinTokensForParallelGather = 64;

namespace onnxruntime {
namespace contrib {

template <typename TScale>
void DequantizeBlockWithMlas(const uint8_t* quantized_data,
                             const TScale* scales,
                             int64_t block_size,
                             int64_t num_bits,
                             int64_t rows,
                             int64_t cols,
                             float* dequantized_data) {
  const float zero_point = num_bits == 8 ? 128.0f : 8.0f;
  const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;

  if (num_bits == 8 && block_size == 0) {
    for (int64_t r = 0; r < rows; ++r) {
      const float scale = static_cast<float>(scales[r]);
      const uint8_t zero_pt = static_cast<uint8_t>(zero_point);

      MlasDequantizeLinear(
          quantized_data + r * cols,
          dequantized_data + r * cols,
          static_cast<size_t>(cols),
          scale,
          zero_pt);
    }
  } else {
    if (num_bits == 8) {
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          const int64_t block_idx = (block_size > 0) ? std::min(c / block_size, blocks_per_row - 1) : 0;
          const int64_t scale_idx = r * blocks_per_row + block_idx;
          const float scale = static_cast<float>(scales[scale_idx]);
          dequantized_data[r * cols + c] = scale * (static_cast<float>(quantized_data[r * cols + c]) - zero_point);
        }
      }
    } else if (num_bits == 4) {
      const int64_t packed_cols = (cols + 1) / 2;
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          const int64_t block_idx = (block_size > 0) ? std::min(c / block_size, blocks_per_row - 1) : 0;
          const int64_t scale_idx = r * blocks_per_row + block_idx;
          const float scale = static_cast<float>(scales[scale_idx]);

          const uint8_t packed_val = quantized_data[r * packed_cols + c / 2];
          const uint8_t quantized_val = (c % 2 == 0) ? (packed_val & 0x0F) : (packed_val >> 4);
          dequantized_data[r * cols + c] = scale * (static_cast<float>(quantized_val) - zero_point);
        }
      }
    }
  }
}

class WorkspacePool {
 public:
  explicit WorkspacePool(AllocatorPtr allocator) : allocator_(allocator) {}

  float* GetWorkspace(size_t size_in_floats) {
    if (size_in_floats <= current_capacity_) {
      return current_workspace_.get();
    }

    size_t new_capacity = std::max(size_in_floats, current_capacity_ * 2);
    current_workspace_ = IAllocator::MakeUniquePtr<float>(allocator_, new_capacity);
    current_capacity_ = new_capacity;
    return current_workspace_.get();
  }

 private:
  AllocatorPtr allocator_;
  IAllocatorUniquePtr<float> current_workspace_;
  size_t current_capacity_ = 0;
};

template <typename TScale>
void DequantizeBlock(const uint8_t* quantized_data,
                     const TScale* scales,
                     int64_t block_size,
                     int64_t num_bits,
                     int64_t rows,
                     int64_t cols,
                     float* dequantized_data) {
  DequantizeBlockWithMlas(quantized_data, scales, block_size, num_bits, rows, cols, dequantized_data);
}

template <typename T>
QMoECPU<T>::QMoECPU(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info),
      MoEBaseCPU(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 4 || expert_weight_bits_ == 8,
              "Attribute 'expert_weight_bits' must be 4 or 8.");
  block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", 0);

  if (block_size_ > 0) {
    ORT_ENFORCE(block_size_ >= 16, "block_size must be >= 16 when provided.");
    ORT_ENFORCE((block_size_ & (block_size_ - 1)) == 0, "block_size must be a power of 2.");
  }
}

template <typename T>
Status QMoECPU<T>::Compute(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  const auto* router_probs = context->Input<Tensor>(1);
  const auto* fc1_experts_weights = context->Input<Tensor>(2);
  const auto* fc1_scales = context->Input<Tensor>(3);
  const auto* fc1_experts_bias = context->Input<Tensor>(4);
  const auto* fc2_experts_weights = context->Input<Tensor>(5);
  const auto* fc2_scales = context->Input<Tensor>(6);
  const auto* fc2_experts_bias = context->Input<Tensor>(7);
  const auto* fc3_experts_weights = context->Input<Tensor>(8);
  const auto* fc3_scales = context->Input<Tensor>(9);
  const auto* fc3_experts_bias = context->Input<Tensor>(10);

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias, fc1_scales,
      fc2_experts_weights, fc2_experts_bias, fc2_scales,
      fc3_experts_weights, fc3_experts_bias, fc3_scales,
      expert_weight_bits_ == 4 ? 2 : 1,
      true,
      block_size_));

  if (fc3_experts_weights || fc3_experts_bias || fc3_scales) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "FC3 gating is not yet implemented on CPU for QMoE");
  }

  const auto& input_shape = input->Shape();
  const int64_t num_tokens = moe_params.num_rows;
  const int64_t hidden_size = moe_params.hidden_size;
  const int64_t inter_size = moe_params.inter_size;
  const int64_t num_experts = moe_params.num_experts;
  const int64_t fc1_out_features = inter_size * (swiglu_fusion_ > 0 ? 2 : 1);

  auto* output = context->Output(0, input_shape);
  auto* tp = context->GetOperatorThreadPool();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const size_t output_buffer_size = static_cast<size_t>(output->Shape().Size());

  const T* input_data = input->Data<T>();

  IAllocatorUniquePtr<float> router_logits_float_buffer;
  const float* router_logits_float;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    router_logits_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * num_experts));
    router_logits_float = router_logits_float_buffer.get();
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(router_probs->Data<T>()),
                                 const_cast<float*>(router_logits_float),
                                 static_cast<size_t>(num_tokens * num_experts));
  } else {
    router_logits_float = reinterpret_cast<const float*>(router_probs->Data<T>());
  }

  auto route_expert_ptr = IAllocator::MakeUniquePtr<int>(allocator, static_cast<size_t>(num_tokens * k_));
  int* route_expert = route_expert_ptr.get();
  auto route_scale_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * k_));
  float* route_scale = route_scale_ptr.get();

  const int num_routing_threads = (tp == nullptr || num_tokens < 512) ? 1 : std::min(static_cast<int>(num_tokens / 64), concurrency::ThreadPool::DegreeOfParallelism(tp));

  std::vector<std::vector<std::vector<int64_t>>> thread_local_expert_token_maps(num_routing_threads);
  for (auto& map : thread_local_expert_token_maps) {
    map.resize(static_cast<size_t>(num_experts));
    for (auto& expert_tokens : map) {
      expert_tokens.reserve(32);
    }
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
      std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + static_cast<std::ptrdiff_t>(k_),
                        sorted_logits.end(), std::greater<>());

      float max_logit = sorted_logits[0].first;

      float sum_exp = 0.0f;
      for (int64_t j = 0; j < k_; ++j) {
        top_k_exp[static_cast<size_t>(j)] = std::exp(sorted_logits[static_cast<size_t>(j)].first - max_logit);
        sum_exp += top_k_exp[static_cast<size_t>(j)];
      }

      const float inv_sum = (sum_exp == 0.0f) ? 0.0f : (1.0f / sum_exp);
      for (int64_t j = 0; j < k_; ++j) {
        int64_t expert_idx = sorted_logits[static_cast<size_t>(j)].second;
        int64_t route_idx = i * k_ + j;
        route_expert[route_idx] = static_cast<int>(expert_idx);
        route_scale[route_idx] = top_k_exp[static_cast<size_t>(j)] * inv_sum;
        if (route_scale[route_idx] > 1e-8f) {  // Use small threshold to avoid zero weights
          local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
        }
      }
    }
  });

  std::vector<std::vector<int64_t>> expert_token_map(static_cast<size_t>(num_experts));
  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    size_t total_tokens_for_expert = 0;
    for (int t = 0; t < num_routing_threads; ++t) {
      total_tokens_for_expert += thread_local_expert_token_maps[t][static_cast<size_t>(expert_idx)].size();
    }
    expert_token_map[static_cast<size_t>(expert_idx)].reserve(total_tokens_for_expert);

    for (int t = 0; t < num_routing_threads; ++t) {
      auto& local_tokens = thread_local_expert_token_maps[t][static_cast<size_t>(expert_idx)];
      if (!local_tokens.empty()) {
        expert_token_map[static_cast<size_t>(expert_idx)].insert(
            expert_token_map[static_cast<size_t>(expert_idx)].end(),
            local_tokens.begin(), local_tokens.end());
      }
    }
  }

  IAllocatorUniquePtr<float> input_float_buffer;
  const float* input_float;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    input_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * hidden_size));
    input_float = input_float_buffer.get();
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(input_data),
                                 const_cast<float*>(input_float),
                                 static_cast<size_t>(num_tokens * hidden_size));
  } else {
    input_float = reinterpret_cast<const float*>(input_data);
  }

  int num_expert_threads = (tp == nullptr) ? 1 : std::min(static_cast<int>(num_experts), concurrency::ThreadPool::DegreeOfParallelism(tp));
  if (num_expert_threads == 0) num_expert_threads = 1;

  auto thread_local_outputs_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_expert_threads) * output_buffer_size);
  float* thread_local_outputs = thread_local_outputs_ptr.get();
  std::memset(thread_local_outputs, 0, static_cast<size_t>(num_expert_threads) * output_buffer_size * sizeof(float));

  size_t max_tokens_per_expert = 0;
  for (const auto& tokens : expert_token_map) {
    max_tokens_per_expert = std::max(max_tokens_per_expert, tokens.size());
  }

  const auto align_size = [](size_t size) -> size_t {
    return (size + kMemoryAlignment - 1) & ~(kMemoryAlignment - 1);
  };

  const size_t A1_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(hidden_size));
  const size_t C1_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(fc1_out_features));
  const size_t A2_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(inter_size));
  const size_t C2_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(hidden_size));
  const size_t B1_dequant_size = align_size(static_cast<size_t>(fc1_out_features) * static_cast<size_t>(hidden_size));
  const size_t B2_dequant_size = align_size(static_cast<size_t>(hidden_size) * static_cast<size_t>(inter_size));
  const size_t bias1_size = align_size(static_cast<size_t>(fc1_out_features));
  const size_t bias2_size = align_size(static_cast<size_t>(hidden_size));

  const size_t workspace_elements_per_thread = A1_size + C1_size + A2_size + C2_size +
                                               B1_dequant_size + B2_dequant_size + bias1_size + bias2_size;

  auto workspace_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_expert_threads) * workspace_elements_per_thread);
  float* workspace = workspace_ptr.get();

  const auto& fc1_scales_dims = fc1_scales->Shape().GetDims();
  const auto& fc2_scales_dims = fc2_scales->Shape().GetDims();
  const bool is_fc1_block_wise = (fc1_scales_dims.size() == 3 && fc1_scales_dims[2] > 1);
  const bool is_fc2_block_wise = (fc2_scales_dims.size() == 3 && fc2_scales_dims[2] > 1);

  const uint8_t* fc1_weights_data = fc1_experts_weights->Data<uint8_t>();
  const uint8_t* fc2_weights_data = fc2_experts_weights->Data<uint8_t>();
  const T* fc1_scales_data = fc1_scales->Data<T>();
  const T* fc2_scales_data = fc2_scales->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias ? fc1_experts_bias->Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias ? fc2_experts_bias->Data<T>() : nullptr;

  concurrency::ThreadPool::TrySimpleParallelFor(tp, num_expert_threads, [&](std::ptrdiff_t thread_id_pd) {
    int thread_id = static_cast<int>(thread_id_pd);
    auto work = concurrency::ThreadPool::PartitionWork(thread_id, num_expert_threads, static_cast<std::ptrdiff_t>(num_experts));

    float* thread_workspace = workspace + static_cast<size_t>(thread_id) * workspace_elements_per_thread;

    for (int64_t expert_idx = work.start; expert_idx < work.end; ++expert_idx) {
      const auto& routes = expert_token_map[static_cast<size_t>(expert_idx)];
      if (routes.empty()) {
        continue;
      }

      const int64_t num_expert_tokens = static_cast<int64_t>(routes.size());

      float* A1 = thread_workspace;
      float* C1 = A1 + A1_size;
      float* A2 = C1 + C1_size;
      float* C2 = A2 + A2_size;
      float* B1_dequant = C2 + C2_size;
      float* B2_dequant = B1_dequant + B1_dequant_size;
      float* bias1_float = B2_dequant + B2_dequant_size;
      float* bias2_float = bias1_float + bias1_size;

      if (static_cast<size_t>(num_expert_tokens) >= kMinTokensForParallelGather) {
        concurrency::ThreadPool::TrySimpleParallelFor(tp, static_cast<int>(num_expert_tokens), [&](std::ptrdiff_t i) {
          const int64_t token_idx = routes[static_cast<size_t>(i)] / k_;
          std::memcpy(A1 + i * hidden_size,
                      input_float + token_idx * hidden_size,
                      static_cast<size_t>(hidden_size) * sizeof(float));
        });
      } else {
        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          const int64_t token_idx = routes[static_cast<size_t>(i)] / k_;
          std::memcpy(A1 + i * hidden_size,
                      input_float + token_idx * hidden_size,
                      static_cast<size_t>(hidden_size) * sizeof(float));
        }
      }

      const T* fc1_scales_ptr;
      if (is_fc1_block_wise) {
        const int64_t fc1_blocks_per_row = fc1_scales_dims[2];
        fc1_scales_ptr = fc1_scales_data + expert_idx * fc1_out_features * fc1_blocks_per_row;
      } else {
        fc1_scales_ptr = fc1_scales_data + expert_idx * fc1_out_features;
      }

      const int64_t pack_unit = (8 / expert_weight_bits_);
      const int64_t fc1_packed_cols = (hidden_size + pack_unit - 1) / pack_unit;

      DequantizeBlock(fc1_weights_data + expert_idx * fc1_out_features * fc1_packed_cols,
                      fc1_scales_ptr,
                      is_fc1_block_wise ? block_size_ : 0, expert_weight_bits_,
                      fc1_out_features, hidden_size, B1_dequant);

      MlasGemm(CblasNoTrans, CblasTrans,
               static_cast<size_t>(num_expert_tokens), static_cast<size_t>(fc1_out_features), static_cast<size_t>(hidden_size),
               1.0f, A1, static_cast<size_t>(hidden_size),
               B1_dequant, static_cast<size_t>(hidden_size),
               0.0f, C1, static_cast<size_t>(fc1_out_features),
               nullptr);

      if (fc1_bias_data) {
        const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), bias1_float, static_cast<size_t>(fc1_out_features));
        } else {
          std::memcpy(bias1_float, B1_bias, static_cast<size_t>(fc1_out_features) * sizeof(float));
        }

        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          float* C1_row = C1 + i * fc1_out_features;
          for (int64_t j = 0; j < fc1_out_features; ++j) {
            C1_row[j] += bias1_float[j];
          }
        }
      }

      for (int64_t i = 0; i < num_expert_tokens; ++i) {
        const float* C1_token = C1 + i * fc1_out_features;
        float* A2_token = A2 + i * inter_size;
        ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
      }

      const T* fc2_scales_ptr;
      if (is_fc2_block_wise) {
        const int64_t fc2_blocks_per_row = fc2_scales_dims[2];
        fc2_scales_ptr = fc2_scales_data + expert_idx * hidden_size * fc2_blocks_per_row;
      } else {
        fc2_scales_ptr = fc2_scales_data + expert_idx * hidden_size;
      }

      const int64_t pack_unit2 = (8 / expert_weight_bits_);
      const int64_t fc2_packed_cols = (inter_size + pack_unit2 - 1) / pack_unit2;

      DequantizeBlock(fc2_weights_data + expert_idx * hidden_size * fc2_packed_cols,
                      fc2_scales_ptr,
                      is_fc2_block_wise ? block_size_ : 0, expert_weight_bits_,
                      hidden_size, inter_size, B2_dequant);

      MlasGemm(CblasNoTrans, CblasTrans,
               static_cast<size_t>(num_expert_tokens), static_cast<size_t>(hidden_size), static_cast<size_t>(inter_size),
               1.0f, A2, static_cast<size_t>(inter_size),
               B2_dequant, static_cast<size_t>(inter_size),
               0.0f, C2, static_cast<size_t>(hidden_size),
               nullptr);

      if (fc2_bias_data) {
        const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), bias2_float, static_cast<size_t>(hidden_size));
        } else {
          std::memcpy(bias2_float, B2_bias, static_cast<size_t>(hidden_size) * sizeof(float));
        }
      }

      for (int64_t i = 0; i < num_expert_tokens; ++i) {
        const int64_t route_idx = routes[static_cast<size_t>(i)];
        const int64_t token_idx = route_idx / k_;
        const float weight = route_scale[route_idx];

        if (token_idx < 0 || token_idx >= num_tokens) continue;

        const size_t buffer_offset = static_cast<size_t>(token_idx) * static_cast<size_t>(hidden_size);
        if (buffer_offset + static_cast<size_t>(hidden_size) > output_buffer_size) {
          continue;
        }

        float* dest = thread_local_outputs + static_cast<size_t>(thread_id) * output_buffer_size + buffer_offset;
        const float* src = C2 + i * hidden_size;

        for (int64_t j = 0; j < hidden_size; ++j) {
          dest[j] += weight * (src[j] + (fc2_bias_data ? bias2_float[j] : 0.0f));
        }
      }
    }
  });
  auto accumulate = [&](float* buffer) {
    std::memset(buffer, 0, output_buffer_size * sizeof(float));

    for (int i = 0; i < num_expert_threads; ++i) {
      const size_t thread_offset = static_cast<size_t>(i) * output_buffer_size;
      const float* src = thread_local_outputs + thread_offset;

      for (size_t j = 0; j < output_buffer_size; ++j) {
        buffer[j] += src[j];
      }
    }
  };

  if constexpr (std::is_same_v<T, MLFloat16>) {
    auto final_output_float_ptr = IAllocator::MakeUniquePtr<float>(allocator, output_buffer_size);
    float* final_output_float = final_output_float_ptr.get();
    accumulate(final_output_float);

    MlasConvertFloatToHalfBuffer(final_output_float,
                                 reinterpret_cast<MLFloat16*>(output->MutableData<T>()),
                                 static_cast<size_t>(output_buffer_size));
  } else {  // T is float
    accumulate(output->MutableData<T>());
  }

  return Status::OK();
}

template QMoECPU<float>::QMoECPU(const OpKernelInfo& op_kernel_info);
template Status QMoECPU<float>::Compute(OpKernelContext* context) const;
template QMoECPU<MLFloat16>::QMoECPU(const OpKernelInfo& op_kernel_info);
template Status QMoECPU<MLFloat16>::Compute(OpKernelContext* context) const;

// Kernel Registration
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE, kMSDomain, 1, float, kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    QMoECPU<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE, kMSDomain, 1, MLFloat16, kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<MLFloat16>()),
    QMoECPU<MLFloat16>);

}  // namespace contrib
}  // namespace onnxruntime
