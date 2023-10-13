// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// #ifdef USE_CUDA

#if 0

#include <memory>

#include "test/util/include/default_providers.h"
#include "cuda_runtime_api.h"

#include "core/common/narrow.h"
#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"

#include "contrib_ops/cuda/bert/paged_attention_impl.h"

namespace onnxruntime {

namespace test {

template <typename ScaleT>
void RefQuantizeReshapeAndCache(
    int num_tokens,
    int num_heads,
    int head_size,
    int x,
    int block_size,
    int num_blocks,
    int kv_quant_chunk_size,
    const std::vector<MLFloat16>& key,       // [num_tokens, num_heads, head_size]
    const std::vector<MLFloat16>& value,     // [num_tokens, num_heads, head_size]
    const std::vector<int>& slot_mapping,    // [num_tokens]
    std::vector<int8_t>& k_cache,            // [num_blocks, num_heads, head_size/x, block_size, x]
    std::vector<int8_t>& v_cache,            // [num_blocks, num_heads, head_size, block_size]
    std::vector<ScaleT>& kv_params_cache) {  // [num_blocks, 2, num_heads, head_size / kv_quant_chunk_size, block_size]
  for (int t = 0; t < num_tokens; t++) {
    for (int h = 0; h < num_heads; h++) {
      for (int i = 0; i < head_size; i += kv_quant_chunk_size) {
        const MLFloat16* k = key.data() + (t * (num_heads * head_size) + h * head_size + i);
        const MLFloat16* v = value.data() + (t * (num_heads * head_size) + h * head_size + i);
        auto max_abs = [](MLFloat16 a, MLFloat16 b) {
          float abs_a = fabsf((float)a);
          float abs_b = fabsf((float)b);
          return MLFloat16(fmaxf(abs_a, abs_b));
        };
        float max_abs_k = (float)std::accumulate(k, k + kv_quant_chunk_size, MLFloat16(0.0f), max_abs);
        float max_abs_v = (float)std::accumulate(v, v + kv_quant_chunk_size, MLFloat16(0.0f), max_abs);
        float inv_unit_scale_k = max_abs_k ? (127.0f / max_abs_k) : 0.0f;
        float inv_unit_scale_v = max_abs_v ? (127.0f / max_abs_v) : 0.0f;

        int s = slot_mapping[t];
        int b = s / block_size;
        int tib = s % block_size;
        auto* base_dst_k = k_cache.data() + ((b * num_heads + h) * head_size * block_size);
        auto* base_dst_v = v_cache.data() + ((b * num_heads + h) * head_size * block_size);
        for (int j = 0; j < kv_quant_chunk_size; j++) {
          int f = i + j;
          auto* dst_k = base_dst_k + ((f / x) * (block_size * x) + tib * x + (f % x));
          auto* dst_v = base_dst_v + (f * block_size + tib);
          *dst_k = (int8_t)std::max(-128, std::min(127, (int)std::nearbyintf((float)k[j] * inv_unit_scale_k)));
          *dst_v = (int8_t)std::max(-128, std::min(127, (int)std::nearbyintf((float)v[j] * inv_unit_scale_v)));
        }

        ScaleT* param_k = kv_params_cache.data() + ((b * 2 * num_heads + h) * (head_size / kv_quant_chunk_size * block_size) + (i / kv_quant_chunk_size) * block_size + tib);
        *param_k = (ScaleT)inv_unit_scale_k;
        ScaleT* param_v = param_k + (num_heads * head_size / kv_quant_chunk_size * block_size);
        *param_v = (ScaleT)inv_unit_scale_v;
      }
    }
  }
}

template <typename T>
std::unique_ptr<T, std::function<void(T*)>> move_to_cuda(const std::vector<T>& cpu_vec) {
  void* dev_ptr = nullptr;
  cudaMalloc(&dev_ptr, sizeof(T) * cpu_vec.size());
  cudaMemcpy(dev_ptr, cpu_vec.data(), sizeof(T) * cpu_vec.size(), cudaMemcpyHostToDevice);
  return std::unique_ptr<T, std::function<void(T*)>>(
      (T*)dev_ptr,
      [](T* p) {
        if (p) {
          cudaFree(p);
          p = nullptr;
        }
      });
}

template <typename T>
void back_to_cpu(const T* dev_ptr, std::vector<T>& cpu_vec) {
  cudaMemcpy(cpu_vec.data(), dev_ptr, sizeof(T) * cpu_vec.size(), cudaMemcpyDeviceToDevice);
}

template <typename ScaleT>
void CudaQuantizeReshapeAndCache(
    int num_tokens,
    int num_heads,
    int head_size,
    int x,
    int block_size,
    int num_blocks,
    int kv_quant_chunk_size,
    const std::vector<MLFloat16>& key,       // [num_tokens, num_heads, head_size]
    const std::vector<MLFloat16>& value,     // [num_tokens, num_heads, head_size]
    const std::vector<int>& slot_mapping,    // [num_tokens]
    std::vector<int8_t>& k_cache,            // [num_blocks, num_heads, head_size/x, block_size, x]
    std::vector<int8_t>& v_cache,            // [num_blocks, num_heads, head_size, block_size]
    std::vector<ScaleT>& kv_params_cache) {  // [num_blocks, 2, num_heads, head_size / kv_quant_chunk_size, block_size]

  auto k_cache_dev = move_to_cuda(k_cache);
  auto v_cache_dev = move_to_cuda(v_cache);
  auto kv_params_dev = move_to_cuda(kv_params_cache);
  auto slot_mapping_dev = move_to_cuda(slot_mapping);
  auto value_dev = move_to_cuda(value);
  auto key_dev = move_to_cuda(key);

  std::vector<int64_t> kv_dims{num_tokens, num_heads, head_size};
  onnxruntime::contrib::cuda::reshape_and_cache(
      nullptr,
      key_dev.get(),
      value_dev.get(),
      k_cache_dev.get(),
      v_cache_dev.get(),
      slot_mapping_dev.get(),
      kv_dims.data(),
      kv_dims.data(),
      block_size,
      x,
      1,
      kv_params_dev.get(),
      kv_quant_chunk_size,
      1);

  ASSERT_TRUE(cudaPeekAtLastError() == cudaSuccess);

  back_to_cpu(k_cache_dev.get(), k_cache);
  back_to_cpu(v_cache_dev.get(), v_cache);
  back_to_cpu(kv_params_dev.get(), kv_params_cache);
}

void check_eq(std::vector<int8_t>& gold, std::vector<int8_t>& r, int threshold = 1) {
  ASSERT_EQ(gold.size(), r.size());
  for (size_t i = 0; i < gold.size(); i++) {
    EXPECT_NEAR(gold[i], r[i], threshold) << " at:" << i;
  }
}

void check_eq(std::vector<MLFloat16>& gold, std::vector<MLFloat16>& r, float abs_err, float rel_err) {
  ASSERT_EQ(gold.size(), r.size());
  for (size_t i = 0; i < gold.size(); i++) {
    EXPECT_NEAR((float)gold[i], (float)r[i], abs_err) << " at:" << i;
    EXPECT_NEAR((float)gold[i], (float)r[i], rel_err * std::fabs((float)gold[i])) << " at:" << i;
  }
}

template <typename ScaleT>
void RunQuantizeReshapeAndCacheKernelTest(
    int num_tokens,
    int num_heads,
    int head_size,
    int x,
    int block_size,
    int kv_quant_chunk_size,
    int seed) {
  assert(num_tokens % block_size == 0);

  int num_blocks = std::min(8 + num_tokens / block_size, 2 * (num_tokens / block_size));
  RandomValueGenerator random{seed};
  std::vector<int64_t> kv_dims{num_blocks, num_heads, head_size};
  std::vector<MLFloat16> key = random.Uniform<MLFloat16>(kv_dims, MLFloat16(-2.0f), MLFloat16(2.0f));
  std::vector<MLFloat16> value = random.Uniform<MLFloat16>(kv_dims, MLFloat16(-2.0f), MLFloat16(2.0f));

  std::vector<int64_t> k_cache_dims{num_blocks, num_heads, head_size / x, block_size, x};
  auto kv_cache_elem_count = std::accumulate(k_cache_dims.begin(), k_cache_dims.end(), 1LL, std::multiplies<int64_t>());
  std::vector<int8_t> k_cache(kv_cache_elem_count, 0);
  std::vector<int8_t> v_cache(kv_cache_elem_count, 0);
  std::vector<int8_t> k_cache_ref(kv_cache_elem_count, 0);
  std::vector<int8_t> v_cache_ref(kv_cache_elem_count, 0);

  std::vector<int> blk_mapping(num_blocks);
  for (auto i = 0; i < num_blocks; i++) blk_mapping[i] = i;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(blk_mapping.begin(), blk_mapping.end(), g);
  std::vector<int> slot_mapping(num_tokens, 0);
  for (auto t = 0; t < num_tokens; t++) {
    int b = blk_mapping[t / block_size];
    slot_mapping[t] = b * block_size + (t % block_size);
  }

  std::vector<int64_t> kv_params_dims{num_blocks, 2, num_heads, head_size / kv_quant_chunk_size, block_size};
  auto kv_params_elem_count = std::accumulate(kv_params_dims.begin(), kv_params_dims.end(), 1LL, std::multiplies<int64_t>());
  std::vector<ScaleT> kv_params_cache(kv_params_elem_count, ScaleT{0.0f});
  std::vector<ScaleT> kv_params_cache_ref(kv_params_elem_count, ScaleT{0.0f});

  RefQuantizeReshapeAndCache(
      num_tokens,
      num_heads,
      head_size,
      x,
      block_size,
      num_blocks,
      kv_quant_chunk_size,
      key,
      value,
      slot_mapping,
      k_cache_ref,
      v_cache_ref,
      kv_params_cache_ref);

  CudaQuantizeReshapeAndCache(
      num_tokens,
      num_heads,
      head_size,
      x,
      block_size,
      num_blocks,
      kv_quant_chunk_size,
      key,
      value,
      slot_mapping,
      k_cache,
      v_cache,
      kv_params_cache);

  check_eq(k_cache_ref, k_cache, 1);
  check_eq(v_cache_ref, v_cache, 1);
  check_eq(kv_params_cache_ref, kv_params_cache, 0.001f, 0.05f);
};

TEST(PagedAttention, QuantizeReshapeAndCache) {
  int num_heads = 3;
  int head_sizes[] = {32, 64, 80, 128};
  int block_sizes[] = {8};
  for (int block_size : block_sizes) {
    for (int head_size : head_sizes) {
      int num_tokens = block_size * 3;
      RunQuantizeReshapeAndCacheKernelTest<MLFloat16>(
          num_tokens,
          num_heads,
          head_size,
          8,  // x
          block_size,
          16,    // kv_quant_chunk_size,
          768);  // seed
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
