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

static constexpr size_t kMemoryAlignment = 64;
static constexpr size_t kMinTokensForParallelGather = 12;
static constexpr int64_t kOptimalBlockSize = 64;
static constexpr int64_t kUnrollFactor = 8;
static constexpr int64_t kCacheLineSize = 64;
static constexpr int64_t kLargeBiasThreshold = 512;

static constexpr size_t kMlasGemmMinBlockSize = 32;
static constexpr size_t kMlasOptimalTileSize = 128;
static constexpr int64_t kDequantBlockAlignment = 16;
static constexpr int64_t kThreadPoolMinWorkSize = 64;
static constexpr int64_t kMlasActivationTileSize = 256;

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
        const uint8_t* row_data = quantized_data + r * cols;
        float* row_output = dequantized_data + r * cols;

        int64_t c = 0;
        if (block_size > 0) {
          for (int64_t block_start = 0; block_start < cols; block_start += block_size) {
            const int64_t block_end = std::min(block_start + block_size, cols);
            const int64_t block_idx = std::min(block_start / block_size, blocks_per_row - 1);
            const int64_t scale_idx = r * blocks_per_row + block_idx;
            const float scale = static_cast<float>(scales[scale_idx]);

            for (c = block_start; c + 4 <= block_end; c += 4) {
              row_output[c] = scale * (static_cast<float>(row_data[c]) - zero_point);
              row_output[c + 1] = scale * (static_cast<float>(row_data[c + 1]) - zero_point);
              row_output[c + 2] = scale * (static_cast<float>(row_data[c + 2]) - zero_point);
              row_output[c + 3] = scale * (static_cast<float>(row_data[c + 3]) - zero_point);
            }
            for (; c < block_end; ++c) {
              row_output[c] = scale * (static_cast<float>(row_data[c]) - zero_point);
            }
          }
        } else {
          const float scale = static_cast<float>(scales[r]);
          for (; c + 8 <= cols; c += 8) {
            row_output[c] = scale * (static_cast<float>(row_data[c]) - zero_point);
            row_output[c + 1] = scale * (static_cast<float>(row_data[c + 1]) - zero_point);
            row_output[c + 2] = scale * (static_cast<float>(row_data[c + 2]) - zero_point);
            row_output[c + 3] = scale * (static_cast<float>(row_data[c + 3]) - zero_point);
            row_output[c + 4] = scale * (static_cast<float>(row_data[c + 4]) - zero_point);
            row_output[c + 5] = scale * (static_cast<float>(row_data[c + 5]) - zero_point);
            row_output[c + 6] = scale * (static_cast<float>(row_data[c + 6]) - zero_point);
            row_output[c + 7] = scale * (static_cast<float>(row_data[c + 7]) - zero_point);
          }
          for (; c < cols; ++c) {
            row_output[c] = scale * (static_cast<float>(row_data[c]) - zero_point);
          }
        }
      }
    } else if (num_bits == 4) {
      const int64_t packed_cols = (cols + 1) / 2;
      for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_data = quantized_data + r * packed_cols;
        float* row_output = dequantized_data + r * cols;

        if (block_size > 0) {
          for (int64_t block_start = 0; block_start < cols; block_start += block_size) {
            const int64_t block_end = std::min(block_start + block_size, cols);
            const int64_t block_idx = std::min(block_start / block_size, blocks_per_row - 1);
            const int64_t scale_idx = r * blocks_per_row + block_idx;
            const float scale = static_cast<float>(scales[scale_idx]);

            for (int64_t c = block_start; c < block_end; c += 2) {
              const uint8_t packed_val = row_data[c / 2];
              const uint8_t val0 = packed_val & 0x0F;
              const uint8_t val1 = packed_val >> 4;

              row_output[c] = scale * (static_cast<float>(val0) - zero_point);
              if (c + 1 < block_end) {
                row_output[c + 1] = scale * (static_cast<float>(val1) - zero_point);
              }
            }
          }
        } else {
          const float scale = static_cast<float>(scales[r]);
          for (int64_t c = 0; c < cols; c += 2) {
            const uint8_t packed_val = row_data[c / 2];
            const uint8_t val0 = packed_val & 0x0F;
            const uint8_t val1 = packed_val >> 4;

            row_output[c] = scale * (static_cast<float>(val0) - zero_point);
            if (c + 1 < cols) {
              row_output[c + 1] = scale * (static_cast<float>(val1) - zero_point);
            }
          }
        }
      }
    }
  }
}

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

  const int max_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;
  const int64_t min_work_per_thread = std::max(kThreadPoolMinWorkSize, static_cast<int64_t>(64));
  const int optimal_routing_threads = (tp == nullptr || num_tokens < min_work_per_thread) ? 1 : std::min(static_cast<int>(num_tokens / min_work_per_thread), max_threads);
  const int num_routing_threads = std::max(1, optimal_routing_threads);

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

  const int max_expert_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;
  const int64_t total_expert_work = std::accumulate(expert_token_map.begin(), expert_token_map.end(), 0LL,
                                                    [](int64_t sum, const std::vector<int64_t>& tokens) { return sum + static_cast<int64_t>(tokens.size()); });
  const int64_t min_expert_work_per_thread = std::max(kThreadPoolMinWorkSize, static_cast<int64_t>(32));

  int num_expert_threads = (tp == nullptr || total_expert_work < min_expert_work_per_thread) ? 1 : std::min(static_cast<int>(total_expert_work / min_expert_work_per_thread), std::min(static_cast<int>(num_experts), max_expert_threads));
  if (num_expert_threads == 0) num_expert_threads = 1;

  auto thread_local_outputs_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_expert_threads) * output_buffer_size);
  float* thread_local_outputs = thread_local_outputs_ptr.get();
  std::memset(thread_local_outputs, 0, static_cast<size_t>(num_expert_threads) * output_buffer_size * sizeof(float));

  size_t max_tokens_per_expert = 0;
  for (const auto& tokens : expert_token_map) {
    max_tokens_per_expert = std::max(max_tokens_per_expert, tokens.size());
  }

  const auto align_size = [](size_t size) -> size_t {
    const size_t mlas_alignment = std::max(kMemoryAlignment, static_cast<size_t>(kMlasOptimalTileSize));
    return (size + mlas_alignment - 1) & ~(mlas_alignment - 1);
  };

  // Pre-compute sizes with enhanced alignment for MLAS optimization
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

  auto bias_conversion_buffers_ptr = IAllocator::MakeUniquePtr<float>(allocator,
                                                                      static_cast<size_t>(num_expert_threads) * (static_cast<size_t>(fc1_out_features) + static_cast<size_t>(hidden_size)));
  float* bias_conversion_buffers = bias_conversion_buffers_ptr.get();

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

  const int64_t pack_unit = (8 / expert_weight_bits_);
  const int64_t fc1_packed_cols = (hidden_size + pack_unit - 1) / pack_unit;
  const int64_t fc2_packed_cols = (inter_size + pack_unit - 1) / pack_unit;
  const bool has_fc1_bias = (fc1_bias_data != nullptr);
  const bool has_fc2_bias = (fc2_bias_data != nullptr);

  std::vector<std::pair<int64_t, size_t>> expert_workload;
  size_t total_work = 0;

  for (int64_t i = 0; i < num_experts; ++i) {
    const size_t token_count = expert_token_map[static_cast<size_t>(i)].size();
    if (token_count > 0) {
      expert_workload.emplace_back(i, token_count);
      total_work += token_count;
    }
  }

  if (total_work < 48) {
    num_expert_threads = 1;
  } else if (total_work < 192) {
    num_expert_threads = std::min(num_expert_threads, 2);
  } else if (total_work < 512) {
    num_expert_threads = std::min(num_expert_threads, 4);
  }

  std::sort(expert_workload.begin(), expert_workload.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  std::vector<std::vector<int64_t>> expert_batches(num_expert_threads);
  size_t thread_idx = 0;
  for (const auto& work : expert_workload) {
    expert_batches[thread_idx].push_back(work.first);
    thread_idx = (thread_idx + 1) % static_cast<size_t>(num_expert_threads);
  }

  concurrency::ThreadPool::TrySimpleParallelFor(tp, num_expert_threads, [&](std::ptrdiff_t thread_id_pd) {
    const int thread_id = static_cast<int>(thread_id_pd);
    const auto& expert_batch = expert_batches[static_cast<size_t>(thread_id)];

    float* thread_workspace = workspace + static_cast<size_t>(thread_id) * workspace_elements_per_thread;

    float* thread_bias1_buffer = bias_conversion_buffers + static_cast<size_t>(thread_id) * (static_cast<size_t>(fc1_out_features) + static_cast<size_t>(hidden_size));
    float* thread_bias2_buffer = thread_bias1_buffer + static_cast<size_t>(fc1_out_features);

    for (int64_t expert_idx : expert_batch) {
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

      const int64_t num_blocks = (num_expert_tokens + kOptimalBlockSize - 1) / kOptimalBlockSize;

      if (static_cast<size_t>(num_expert_tokens) >= kMinTokensForParallelGather && num_blocks > 2) {
        concurrency::ThreadPool::TrySimpleParallelFor(tp, static_cast<int>(num_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_idx = block_idx * kOptimalBlockSize;
          const int64_t end_idx = std::min(start_idx + kOptimalBlockSize, num_expert_tokens);

          for (int64_t i = start_idx; i < end_idx; ++i) {
            const int64_t token_idx = routes[static_cast<size_t>(i)] / k_;
            const float* src = input_float + token_idx * hidden_size;
            float* dst = A1 + i * hidden_size;

            std::memcpy(dst, src, static_cast<size_t>(hidden_size) * sizeof(float));
          }
        });
      } else {
        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          const int64_t token_idx = routes[static_cast<size_t>(i)] / k_;
          const float* src = input_float + token_idx * hidden_size;
          float* dst = A1 + i * hidden_size;

          if (hidden_size >= 64) {
            std::memcpy(dst, src, static_cast<size_t>(hidden_size) * sizeof(float));
          } else {
            int64_t j = 0;
            for (; j + kUnrollFactor <= hidden_size; j += kUnrollFactor) {
              dst[j] = src[j];
              dst[j + 1] = src[j + 1];
              dst[j + 2] = src[j + 2];
              dst[j + 3] = src[j + 3];
              dst[j + 4] = src[j + 4];
              dst[j + 5] = src[j + 5];
              dst[j + 6] = src[j + 6];
              dst[j + 7] = src[j + 7];
            }
            for (; j < hidden_size; ++j) {
              dst[j] = src[j];
            }
          }
        }
      }

      const T* fc1_scales_ptr;

      if (is_fc1_block_wise) {
        const int64_t fc1_blocks_per_row = fc1_scales_dims[2];
        fc1_scales_ptr = fc1_scales_data + expert_idx * fc1_out_features * fc1_blocks_per_row;
      } else {
        fc1_scales_ptr = fc1_scales_data + expert_idx * fc1_out_features;
      }

      const int64_t dequant_block_size = std::max(kDequantBlockAlignment,
                                                  std::min(static_cast<int64_t>(kMlasOptimalTileSize),
                                                           fc1_out_features / 4));
      const int64_t num_dequant_blocks = (fc1_out_features + dequant_block_size - 1) / dequant_block_size;

      if (num_dequant_blocks > 1 && fc1_out_features >= static_cast<int64_t>(kMlasGemmMinBlockSize)) {
        concurrency::ThreadPool::TrySimpleParallelFor(tp, static_cast<int>(num_dequant_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_row = block_idx * dequant_block_size;
          const int64_t end_row = std::min(start_row + dequant_block_size, fc1_out_features);
          const auto offset = expert_idx * fc1_out_features * fc1_packed_cols + start_row * fc1_packed_cols;
          DequantizeBlock(fc1_weights_data + offset,
                          fc1_scales_ptr + (is_fc1_block_wise ? start_row * fc1_scales_dims[2] : start_row),
                          is_fc1_block_wise ? block_size_ : 0, expert_weight_bits_,
                          end_row - start_row, hidden_size, B1_dequant + start_row * hidden_size);
        });
      } else {
        DequantizeBlock(fc1_weights_data + expert_idx * fc1_out_features * fc1_packed_cols,
                        fc1_scales_ptr,
                        is_fc1_block_wise ? block_size_ : 0, expert_weight_bits_,
                        fc1_out_features, hidden_size, B1_dequant);
      }

      // Optimized MLAS GEMM call with better blocking for cache efficiency
      const size_t m = static_cast<size_t>(num_expert_tokens);
      const size_t n = static_cast<size_t>(fc1_out_features);
      const size_t k = static_cast<size_t>(hidden_size);

      // Use optimized MLAS parameters for better performance
      MlasGemm(CblasNoTrans, CblasTrans,
               m, n, k,
               1.0f, A1, k,
               B1_dequant, k,
               0.0f, C1, n,
               tp);

      if (has_fc1_bias) {
        const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), thread_bias1_buffer, static_cast<size_t>(fc1_out_features));
        } else {
          if (fc1_out_features >= kLargeBiasThreshold) {
            std::memcpy(thread_bias1_buffer, B1_bias, static_cast<size_t>(fc1_out_features) * sizeof(float));
          } else {
            int64_t j = 0;
            for (; j + kUnrollFactor <= fc1_out_features; j += kUnrollFactor) {
              thread_bias1_buffer[j] = static_cast<float>(B1_bias[j]);
              thread_bias1_buffer[j + 1] = static_cast<float>(B1_bias[j + 1]);
              thread_bias1_buffer[j + 2] = static_cast<float>(B1_bias[j + 2]);
              thread_bias1_buffer[j + 3] = static_cast<float>(B1_bias[j + 3]);
              thread_bias1_buffer[j + 4] = static_cast<float>(B1_bias[j + 4]);
              thread_bias1_buffer[j + 5] = static_cast<float>(B1_bias[j + 5]);
              thread_bias1_buffer[j + 6] = static_cast<float>(B1_bias[j + 6]);
              thread_bias1_buffer[j + 7] = static_cast<float>(B1_bias[j + 7]);
            }
            for (; j < fc1_out_features; ++j) {
              thread_bias1_buffer[j] = static_cast<float>(B1_bias[j]);
            }
          }
        }

        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          float* C1_row = C1 + i * fc1_out_features;

          int64_t j = 0;
          for (; j + kUnrollFactor <= fc1_out_features; j += kUnrollFactor) {
            C1_row[j] += thread_bias1_buffer[j];
            C1_row[j + 1] += thread_bias1_buffer[j + 1];
            C1_row[j + 2] += thread_bias1_buffer[j + 2];
            C1_row[j + 3] += thread_bias1_buffer[j + 3];
            C1_row[j + 4] += thread_bias1_buffer[j + 4];
            C1_row[j + 5] += thread_bias1_buffer[j + 5];
            C1_row[j + 6] += thread_bias1_buffer[j + 6];
            C1_row[j + 7] += thread_bias1_buffer[j + 7];
          }
          for (; j < fc1_out_features; ++j) {
            C1_row[j] += thread_bias1_buffer[j];
          }
        }
      }

      if (num_expert_tokens >= static_cast<int64_t>(kMlasActivationTileSize / inter_size) && tp != nullptr) {
        const int64_t activation_block_size = std::max(int64_t{1}, kMlasActivationTileSize / inter_size);
        const int64_t num_activation_blocks = (num_expert_tokens + activation_block_size - 1) / activation_block_size;

        if (num_activation_blocks > 1) {
          concurrency::ThreadPool::TrySimpleParallelFor(tp, static_cast<int>(num_activation_blocks), [&](std::ptrdiff_t block_idx) {
            const int64_t start_token = block_idx * activation_block_size;
            const int64_t end_token = std::min(start_token + activation_block_size, num_expert_tokens);

            for (int64_t i = start_token; i < end_token; ++i) {
              const float* C1_token = C1 + i * fc1_out_features;
              float* A2_token = A2 + i * inter_size;
              ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
            }
          });
        } else {
          for (int64_t i = 0; i < num_expert_tokens; ++i) {
            const float* C1_token = C1 + i * fc1_out_features;
            float* A2_token = A2 + i * inter_size;
            ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
          }
        }
      } else {
        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          const float* C1_token = C1 + i * fc1_out_features;
          float* A2_token = A2 + i * inter_size;
          ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
        }
      }

      const T* fc2_scales_ptr;

      if (is_fc2_block_wise) {
        const int64_t fc2_blocks_per_row = fc2_scales_dims[2];
        fc2_scales_ptr = fc2_scales_data + expert_idx * hidden_size * fc2_blocks_per_row;
      } else {
        fc2_scales_ptr = fc2_scales_data + expert_idx * hidden_size;
      }

      const int64_t fc2_dequant_block_size = std::max(kDequantBlockAlignment,
                                                      std::min(static_cast<int64_t>(kMlasOptimalTileSize / 2),
                                                               hidden_size / 4));
      const int64_t num_fc2_dequant_blocks = (hidden_size + fc2_dequant_block_size - 1) / fc2_dequant_block_size;

      if (num_fc2_dequant_blocks > 1 && hidden_size >= static_cast<int64_t>(kMlasGemmMinBlockSize)) {
        concurrency::ThreadPool::TrySimpleParallelFor(tp, static_cast<int>(num_fc2_dequant_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_row = block_idx * fc2_dequant_block_size;
          const int64_t end_row = std::min(start_row + fc2_dequant_block_size, hidden_size);
          const auto offset = expert_idx * hidden_size * fc2_packed_cols + start_row * fc2_packed_cols;
          DequantizeBlock(fc2_weights_data + offset,
                          fc2_scales_ptr + (is_fc2_block_wise ? start_row * fc2_scales_dims[2] : start_row),
                          is_fc2_block_wise ? block_size_ : 0, expert_weight_bits_,
                          end_row - start_row, inter_size, B2_dequant + start_row * inter_size);
        });
      } else {
        DequantizeBlock(fc2_weights_data + expert_idx * hidden_size * fc2_packed_cols,
                        fc2_scales_ptr,
                        is_fc2_block_wise ? block_size_ : 0, expert_weight_bits_,
                        hidden_size, inter_size, B2_dequant);
      }

      const size_t m2 = static_cast<size_t>(num_expert_tokens);
      const size_t n2 = static_cast<size_t>(hidden_size);
      const size_t k2 = static_cast<size_t>(inter_size);

      MlasGemm(CblasNoTrans, CblasTrans,
               m2, n2, k2,
               1.0f, A2, k2,
               B2_dequant, k2,
               0.0f, C2, n2,
               tp);

      if (has_fc2_bias) {
        const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), thread_bias2_buffer, static_cast<size_t>(hidden_size));
        } else {
          if (hidden_size >= kLargeBiasThreshold) {
            std::memcpy(thread_bias2_buffer, B2_bias, static_cast<size_t>(hidden_size) * sizeof(float));
          } else {
            int64_t j = 0;
            for (; j + kUnrollFactor <= hidden_size; j += kUnrollFactor) {
              thread_bias2_buffer[j] = static_cast<float>(B2_bias[j]);
              thread_bias2_buffer[j + 1] = static_cast<float>(B2_bias[j + 1]);
              thread_bias2_buffer[j + 2] = static_cast<float>(B2_bias[j + 2]);
              thread_bias2_buffer[j + 3] = static_cast<float>(B2_bias[j + 3]);
              thread_bias2_buffer[j + 4] = static_cast<float>(B2_bias[j + 4]);
              thread_bias2_buffer[j + 5] = static_cast<float>(B2_bias[j + 5]);
              thread_bias2_buffer[j + 6] = static_cast<float>(B2_bias[j + 6]);
              thread_bias2_buffer[j + 7] = static_cast<float>(B2_bias[j + 7]);
            }
            for (; j < hidden_size; ++j) {
              thread_bias2_buffer[j] = static_cast<float>(B2_bias[j]);
            }
          }
        }
      }

      for (int64_t i = 0; i < num_expert_tokens; ++i) {
        const int64_t route_idx = routes[static_cast<size_t>(i)];
        const int64_t token_idx = route_idx / k_;
        const float weight = route_scale[route_idx];

        if (token_idx < 0 || token_idx >= num_tokens) continue;

        const size_t buffer_offset = static_cast<size_t>(token_idx) * static_cast<size_t>(hidden_size);
        if (buffer_offset + static_cast<size_t>(hidden_size) > output_buffer_size) continue;

        float* dest = thread_local_outputs + static_cast<size_t>(thread_id) * output_buffer_size + buffer_offset;
        const float* src = C2 + i * hidden_size;

        if (has_fc2_bias) {
          int64_t j = 0;
          for (; j + kUnrollFactor <= hidden_size; j += kUnrollFactor) {
            dest[j] += weight * (src[j] + thread_bias2_buffer[j]);
            dest[j + 1] += weight * (src[j + 1] + thread_bias2_buffer[j + 1]);
            dest[j + 2] += weight * (src[j + 2] + thread_bias2_buffer[j + 2]);
            dest[j + 3] += weight * (src[j + 3] + thread_bias2_buffer[j + 3]);
            dest[j + 4] += weight * (src[j + 4] + thread_bias2_buffer[j + 4]);
            dest[j + 5] += weight * (src[j + 5] + thread_bias2_buffer[j + 5]);
            dest[j + 6] += weight * (src[j + 6] + thread_bias2_buffer[j + 6]);
            dest[j + 7] += weight * (src[j + 7] + thread_bias2_buffer[j + 7]);
          }
          for (; j < hidden_size; ++j) {
            dest[j] += weight * (src[j] + thread_bias2_buffer[j]);
          }
        } else {
          int64_t j = 0;
          for (; j + kUnrollFactor <= hidden_size; j += kUnrollFactor) {
            dest[j] += weight * src[j];
            dest[j + 1] += weight * src[j + 1];
            dest[j + 2] += weight * src[j + 2];
            dest[j + 3] += weight * src[j + 3];
            dest[j + 4] += weight * src[j + 4];
            dest[j + 5] += weight * src[j + 5];
            dest[j + 6] += weight * src[j + 6];
            dest[j + 7] += weight * src[j + 7];
          }
          for (; j < hidden_size; ++j) {
            dest[j] += weight * src[j];
          }
        }
      }
    }
  });

  auto accumulate = [&](float* buffer) {
    std::memset(buffer, 0, output_buffer_size * sizeof(float));

    const int max_acc_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;
    const int64_t min_elements_per_thread = std::max(kThreadPoolMinWorkSize * 8, static_cast<int64_t>(256));
    const int optimal_acc_threads = (tp == nullptr || output_buffer_size < min_elements_per_thread) ? 1 : std::min(static_cast<int>(output_buffer_size / min_elements_per_thread), max_acc_threads);
    const int num_acc_threads = std::max(1, optimal_acc_threads);

    if (num_acc_threads > 1) {
      concurrency::ThreadPool::TrySimpleParallelFor(tp, num_acc_threads, [&](std::ptrdiff_t acc_thread_id) {
        const size_t elements_per_thread = output_buffer_size / static_cast<size_t>(num_acc_threads);
        const size_t start_idx = static_cast<size_t>(acc_thread_id) * elements_per_thread;
        const size_t end_idx = (acc_thread_id == num_acc_threads - 1) ? output_buffer_size : start_idx + elements_per_thread;

        for (int i = 0; i < num_expert_threads; ++i) {
          const size_t thread_offset = static_cast<size_t>(i) * output_buffer_size;
          const float* src = thread_local_outputs + thread_offset + start_idx;
          float* dst = buffer + start_idx;

          size_t j = 0;
          const size_t chunk_size = end_idx - start_idx;
          for (; j + kUnrollFactor <= chunk_size; j += kUnrollFactor) {
            dst[j] += src[j];
            dst[j + 1] += src[j + 1];
            dst[j + 2] += src[j + 2];
            dst[j + 3] += src[j + 3];
            dst[j + 4] += src[j + 4];
            dst[j + 5] += src[j + 5];
            dst[j + 6] += src[j + 6];
            dst[j + 7] += src[j + 7];
          }
          for (; j < chunk_size; ++j) {
            dst[j] += src[j];
          }
        }
      });
    } else {
      for (int i = 0; i < num_expert_threads; ++i) {
        const size_t thread_offset = static_cast<size_t>(i) * output_buffer_size;
        const float* src = thread_local_outputs + thread_offset;

        size_t j = 0;
        for (; j + kUnrollFactor <= output_buffer_size; j += kUnrollFactor) {
          buffer[j] += src[j];
          buffer[j + 1] += src[j + 1];
          buffer[j + 2] += src[j + 2];
          buffer[j + 3] += src[j + 3];
          buffer[j + 4] += src[j + 4];
          buffer[j + 5] += src[j + 5];
          buffer[j + 6] += src[j + 6];
          buffer[j + 7] += src[j + 7];
        }
        for (; j < output_buffer_size; ++j) {
          buffer[j] += src[j];
        }
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
