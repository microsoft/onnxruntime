// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// OpTest: Exercises GroupQueryAttention (GQA) on WebGPU EP with flash attention.
//
// Generates deterministic fp16 inputs, runs GQA with a static KV cache, and validates:
//   1. Correctness: output is finite and non-zero
//   2. KV cache: present_key/value contain the expected values at the written positions
//   3. Performance: measures latency for prompt and decode across cache lengths 256, 1024, 4096
//
// Generate the model first:  python generate_gqa_model.py
// Build:  see CMakeLists.txt
// Run:    OpTest.exe [gqa_model.onnx]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

#define THROW_IF_NOT(cond)                                       \
  do {                                                           \
    if (!(cond)) {                                               \
      throw std::runtime_error(std::string(__FILE__) + ":" +     \
                               std::to_string(__LINE__) + ": " + \
                               "check failed: " #cond);          \
    }                                                            \
  } while (0)

// Minimal fp16 <-> fp32 conversion (IEEE 754 half-precision).
static uint16_t fp32_to_fp16(float value) {
  uint32_t f;
  std::memcpy(&f, &value, sizeof(f));
  uint32_t sign = (f >> 16) & 0x8000;
  int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = f & 0x7FFFFF;
  if (exponent <= 0) {
    return static_cast<uint16_t>(sign);  // flush to zero
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00);  // infinity
  }
  return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
}

static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x03FF;
  if (exponent == 0) {
    if (mantissa == 0) {
      uint32_t result = sign;
      float out;
      std::memcpy(&out, &result, sizeof(out));
      return out;
    }
    // subnormal
    exponent = 1;
    while (!(mantissa & 0x0400)) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x03FF;
    exponent = exponent + 127 - 15;
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float out;
    std::memcpy(&out, &result, sizeof(out));
    return out;
  }
  if (exponent == 31) {
    uint32_t result = sign | 0x7F800000 | (mantissa << 13);
    float out;
    std::memcpy(&out, &result, sizeof(out));
    return out;
  }
  uint32_t result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  float out;
  std::memcpy(&out, &result, sizeof(out));
  return out;
}

// Fill fp16 buffer with Gaussian-distributed values (mean=0, std=0.5).
// Real LLM Q/K/V are linear projections of LayerNorm'd embeddings —
// approximately Gaussian with moderate variance, NOT uniformly random.
static void fill_random_fp16(uint16_t* data, size_t count, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 0.5f);
  for (size_t i = 0; i < count; ++i) {
    float v = dist(rng);
    // Clamp to fp16 range to avoid inf
    v = std::max(-65504.0f, std::min(65504.0f, v));
    data[i] = fp32_to_fp16(v);
  }
}

// Fill fp16 buffer with zeros
static void fill_zero_fp16(uint16_t* data, size_t count) {
  std::memset(data, 0, count * sizeof(uint16_t));
}

// Check that fp16 buffer contains finite values and not all zeros
static bool check_output_valid(const uint16_t* data, size_t count) {
  bool has_nonzero = false;
  for (size_t i = 0; i < count; ++i) {
    float v = fp16_to_fp32(data[i]);
    if (!std::isfinite(v)) return false;
    if (v != 0.0f) has_nonzero = true;
  }
  return has_nonzero;
}

// Compute L2 norm of fp16 buffer
static double compute_l2_norm(const uint16_t* data, size_t count) {
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double v = fp16_to_fp32(data[i]);
    sum += v * v;
  }
  return std::sqrt(sum);
}

// --------------------------------------------------------------------------
// CPU Reference: Fast Walsh-Hadamard Transform (FWHT)
// --------------------------------------------------------------------------

// Apply normalized FWHT in-place to a single fp16 vector of `head_size` elements.
// This is the CPU reference equivalent of the TurboQuantRotate GPU shader.
// The transform is self-inverse: applying it twice returns the original vector.
static void cpu_fwht_fp16(uint16_t* vec, int head_size) {
  // Work in float for precision
  std::vector<float> buf(head_size);
  for (int i = 0; i < head_size; ++i) {
    buf[i] = fp16_to_fp32(vec[i]);
  }

  // Iterative butterfly FWHT (same algorithm as the WGSL shader)
  for (int half_block = 1; half_block < head_size; half_block <<= 1) {
    int block_size = half_block << 1;
    for (int block_start = 0; block_start < head_size; block_start += block_size) {
      for (int j = 0; j < half_block; ++j) {
        int i0 = block_start + j;
        int i1 = i0 + half_block;
        float a = buf[i0];
        float b = buf[i1];
        buf[i0] = a + b;
        buf[i1] = a - b;
      }
    }
  }

  // Normalize by 1/sqrt(head_size)
  float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (int i = 0; i < head_size; ++i) {
    vec[i] = fp32_to_fp16(buf[i] * scale);
  }
}

// Apply FWHT to specified token range in a BNSH tensor on CPU.
// Shape: [batch, num_heads, max_seq, head_size]
// Rotates tokens in [start_token, start_token + num_tokens) for all batch/head combos.
static void cpu_rotate_bnsh(uint16_t* data, int batch_size, int num_heads,
                            int max_seq, int head_size,
                            int start_token, int num_tokens) {
  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      for (int t = 0; t < num_tokens; ++t) {
        size_t offset = (static_cast<size_t>(b) * num_heads + h) * max_seq * head_size +
                        static_cast<size_t>(start_token + t) * head_size;
        cpu_fwht_fp16(data + offset, head_size);
      }
    }
  }
}

// --------------------------------------------------------------------------
// CPU Reference: TurboQuant Pseudo-Quantization
// --------------------------------------------------------------------------

static const float TQ_CENTROIDS[16] = {
    -0.2377f, -0.1809f, -0.1419f, -0.1104f, -0.0829f, -0.0578f, -0.0342f, -0.0113f,
     0.0113f,  0.0342f,  0.0578f,  0.0829f,  0.1104f,  0.1419f,  0.1809f,  0.2377f};

static const float TQ_BOUNDARIES[15] = {
    -0.2093f, -0.1614f, -0.1261f, -0.0966f, -0.0704f, -0.0460f, -0.0227f,
     0.0000f,  0.0227f,  0.0460f,  0.0704f,  0.0966f,  0.1261f,  0.1614f,  0.2093f};

// Binary search over 15 boundaries → returns index 0..15
static uint32_t cpu_searchsorted_tq(float val) {
  uint32_t lo = 0, hi = 15;
  for (int iter = 0; iter < 4; ++iter) {
    uint32_t mid = (lo + hi) >> 1;
    if (val >= TQ_BOUNDARIES[mid]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Apply fused FWHT rotation + pseudo-quantization to a single head vector.
// Input: fp16 vector of head_size elements (source).
// Output: compressed packed format written to dest (compressed_dim fp16 elements).
//   [0..1]: f32 norm as bitcast<vec2<fp16>>
//   [2..compressed_dim-3]: int4 indices packed 8 per u32 (each u32 stored as 2 fp16)
//   [compressed_dim-2..compressed_dim-1]: padding zeros
static void cpu_fwht_quantize_packed(const uint16_t* src, uint16_t* dest,
                                     int head_size, int compressed_dim) {
  // Load to f32
  std::vector<float> buf(head_size);
  for (int i = 0; i < head_size; ++i) {
    buf[i] = fp16_to_fp32(src[i]);
  }

  // FWHT in f32
  for (int half_block = 1; half_block < head_size; half_block <<= 1) {
    int block_size = half_block << 1;
    for (int block_start = 0; block_start < head_size; block_start += block_size) {
      for (int j = 0; j < half_block; ++j) {
        int i0 = block_start + j;
        int i1 = i0 + half_block;
        float a = buf[i0];
        float b = buf[i1];
        buf[i0] = a + b;
        buf[i1] = a - b;
      }
    }
  }
  float fwht_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (int i = 0; i < head_size; ++i) {
    buf[i] *= fwht_scale;
  }

  // Compute L2 norm in f32 (ALL head_size elements)
  double sum_sq = 0.0;
  for (int i = 0; i < head_size; ++i) {
    sum_sq += static_cast<double>(buf[i]) * buf[i];
  }
  float norm = static_cast<float>(std::sqrt(std::max(sum_sq, 1e-12)));
  float inv_norm = 1.0f / norm;

  // Quantize all head_size elements to 4-bit indices
  std::vector<uint32_t> indices(head_size);
  for (int i = 0; i < head_size; ++i) {
    float unit_val = buf[i] * inv_norm;
    indices[i] = cpu_searchsorted_tq(unit_val);
  }

  // Clear dest
  std::memset(dest, 0, compressed_dim * sizeof(uint16_t));

  // Store f32 norm in first 2 fp16 slots via bitcast
  uint32_t norm_bits;
  std::memcpy(&norm_bits, &norm, sizeof(norm_bits));
  dest[0] = static_cast<uint16_t>(norm_bits & 0xFFFF);
  dest[1] = static_cast<uint16_t>((norm_bits >> 16) & 0xFFFF);

  // Pack 8 indices per u32, store as 2 fp16 values starting at slot 2
  int n_groups = head_size / 8;
  for (int g = 0; g < n_groups; ++g) {
    uint32_t packed = 0;
    for (int k = 0; k < 8; ++k) {
      packed |= (indices[g * 8 + k] & 0xF) << (k * 4);
    }
    uint16_t lo16, hi16;
    std::memcpy(&lo16, reinterpret_cast<const char*>(&packed), 2);
    std::memcpy(&hi16, reinterpret_cast<const char*>(&packed) + 2, 2);
    dest[2 + g * 2] = lo16;
    dest[2 + g * 2 + 1] = hi16;
  }
}

// Apply fused FWHT + quantize + pack to a token range.
// Reads from src BNSH tensor (head_size per token), writes to dest BNSH (compressed_dim per token).
static void cpu_rotate_quantize_packed_bnsh(const uint16_t* src, uint16_t* dest,
                                            int batch_size, int num_heads,
                                            int max_seq, int head_size, int compressed_dim,
                                            int start_token, int num_tokens) {
  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      for (int t = 0; t < num_tokens; ++t) {
        size_t src_offset = (static_cast<size_t>(b) * num_heads + h) * max_seq * head_size +
                            static_cast<size_t>(start_token + t) * head_size;
        size_t dest_offset = (static_cast<size_t>(b) * num_heads + h) * max_seq * compressed_dim +
                             static_cast<size_t>(start_token + t) * compressed_dim;
        cpu_fwht_quantize_packed(src + src_offset, dest + dest_offset, head_size, compressed_dim);
      }
    }
  }
}

// Read f32 norm from the first 2 fp16 slots of a packed compressed vector.
static float read_packed_norm(const uint16_t* packed_token) {
  uint32_t lo = packed_token[0];
  uint32_t hi = packed_token[1];
  uint32_t bits = lo | (hi << 16);
  float norm;
  std::memcpy(&norm, &bits, sizeof(norm));
  return norm;
}

// Extract a 4-bit index from packed compressed format.
// element_idx: element index within the head (0..head_size-1)
static uint32_t extract_packed_index(const uint16_t* packed_token, int element_idx) {
  int u32_group = element_idx / 8;
  int bit_pos = (element_idx % 8) * 4;
  // The u32 is stored at fp16 slots [2 + u32_group*2, 2 + u32_group*2 + 1]
  uint32_t packed_u32;
  std::memcpy(&packed_u32, &packed_token[2 + u32_group * 2], sizeof(uint32_t));
  return (packed_u32 >> bit_pos) & 0xF;
}

// Compare centroid indices between two packed compressed BNSH tensors.
static double compare_packed_indices(const uint16_t* a, const uint16_t* b,
                                     int batch_size, int num_heads,
                                     int max_seq, int head_size, int compressed_dim,
                                     int start_token, int num_tokens) {
  size_t total = 0, matches = 0;
  for (int bx = 0; bx < batch_size; ++bx) {
    for (int h = 0; h < num_heads; ++h) {
      for (int t = 0; t < num_tokens; ++t) {
        size_t offset = (static_cast<size_t>(bx) * num_heads + h) * max_seq * compressed_dim +
                        static_cast<size_t>(start_token + t) * compressed_dim;
        const uint16_t* a_token = a + offset;
        const uint16_t* b_token = b + offset;
        for (int i = 0; i < head_size; ++i) {
          uint32_t idx_a = extract_packed_index(a_token, i);
          uint32_t idx_b = extract_packed_index(b_token, i);
          if (idx_a == idx_b) ++matches;
          ++total;
        }
      }
    }
  }
  return (total > 0) ? static_cast<double>(matches) / total : 1.0;
}

// --------------------------------------------------------------------------
// GQA Test Configuration
// --------------------------------------------------------------------------
struct GqaConfig {
  int batch_size = 1;
  int num_heads = 24;
  int kv_num_heads = 8;
  int head_size = 128;
  int max_cache = 131072;  // must match model's max_cache
  int compressed_dim = 0;  // head_size/4 + 4 when turbo_quant (vec4-aligned)

  void init_compressed_dim() {
    compressed_dim = ((head_size / 4 + 2 + 3) / 4) * 4;
  }

  int cache_last_dim(bool turbo_quant) const {
    return turbo_quant ? compressed_dim : head_size;
  }
};

// --------------------------------------------------------------------------
// Run a single GQA inference
// --------------------------------------------------------------------------
struct GqaResult {
  std::vector<uint16_t> output;
  std::vector<uint16_t> present_key;
  std::vector<uint16_t> present_value;
  double elapsed_ms;
};

GqaResult run_gqa(Ort::Session& session,
                  const GqaConfig& cfg,
                  int seq_len,
                  int past_seq_len,
                  const uint16_t* past_key_data,
                  const uint16_t* past_value_data,
                  std::mt19937& rng,
                  bool turbo_quant = false,
                  int past_cache_dim_override = 0,
                  int num_iters = 3) {
  const int hidden_size = cfg.num_heads * cfg.head_size;
  const int kv_hidden_size = cfg.kv_num_heads * cfg.head_size;
  const int total_seq = past_seq_len + seq_len;
  // Use override if provided (for compressed TQ past), otherwise head_size
  const int cache_dim = past_cache_dim_override > 0 ? past_cache_dim_override : cfg.head_size;

  // Allocate inputs
  size_t q_size = cfg.batch_size * seq_len * hidden_size;
  size_t k_size = cfg.batch_size * seq_len * kv_hidden_size;
  size_t v_size = k_size;
  size_t cache_size = static_cast<size_t>(cfg.batch_size) * cfg.kv_num_heads * cfg.max_cache * cache_dim;

  std::vector<uint16_t> query_data(q_size);
  std::vector<uint16_t> key_data(k_size);
  std::vector<uint16_t> value_data(v_size);
  std::vector<uint16_t> past_key(cache_size);
  std::vector<uint16_t> past_value(cache_size);

  fill_random_fp16(query_data.data(), q_size, rng);
  fill_random_fp16(key_data.data(), k_size, rng);
  fill_random_fp16(value_data.data(), v_size, rng);

  // Copy past KV data if provided
  if (past_key_data) std::memcpy(past_key.data(), past_key_data, cache_size * sizeof(uint16_t));
  if (past_value_data) std::memcpy(past_value.data(), past_value_data, cache_size * sizeof(uint16_t));

  // seqlens_k = total_seq - 1 (per batch)
  std::vector<int32_t> seqlens_k_data(cfg.batch_size, total_seq - 1);
  // total_sequence_length scalar
  std::vector<int32_t> total_seq_data = {static_cast<int32_t>(total_seq)};

  // Create ORT tensors
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 3> q_shape = {cfg.batch_size, seq_len, hidden_size};
  std::array<int64_t, 3> k_shape = {cfg.batch_size, seq_len, kv_hidden_size};
  std::array<int64_t, 3> v_shape = {cfg.batch_size, seq_len, kv_hidden_size};
  std::array<int64_t, 4> cache_shape = {cfg.batch_size, cfg.kv_num_heads,
                                         static_cast<int64_t>(cfg.max_cache),
                                         cache_dim};
  std::array<int64_t, 1> seqlens_shape = {cfg.batch_size};
  std::array<int64_t, 1> total_seq_shape = {1};

  auto q_tensor = Ort::Value::CreateTensor(
      mem_info, query_data.data(), q_size * sizeof(uint16_t),
      q_shape.data(), q_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  auto k_tensor = Ort::Value::CreateTensor(
      mem_info, key_data.data(), k_size * sizeof(uint16_t),
      k_shape.data(), k_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  auto v_tensor = Ort::Value::CreateTensor(
      mem_info, value_data.data(), v_size * sizeof(uint16_t),
      v_shape.data(), v_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  auto past_key_tensor = Ort::Value::CreateTensor(
      mem_info, past_key.data(), cache_size * sizeof(uint16_t),
      cache_shape.data(), cache_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  auto past_value_tensor = Ort::Value::CreateTensor(
      mem_info, past_value.data(), cache_size * sizeof(uint16_t),
      cache_shape.data(), cache_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  auto seqlens_tensor = Ort::Value::CreateTensor(
      mem_info, seqlens_k_data.data(), seqlens_k_data.size() * sizeof(int32_t),
      seqlens_shape.data(), seqlens_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto total_seq_tensor = Ort::Value::CreateTensor(
      mem_info, total_seq_data.data(), total_seq_data.size() * sizeof(int32_t),
      total_seq_shape.data(), total_seq_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

  // Input/output names
  const char* input_names[] = {"query", "key", "value", "past_key", "past_value",
                               "seqlens_k", "total_sequence_length"};
  const char* output_names[] = {"output", "present_key", "present_value"};

  std::array<Ort::Value, 7> inputs;
  inputs[0] = std::move(q_tensor);
  inputs[1] = std::move(k_tensor);
  inputs[2] = std::move(v_tensor);
  inputs[3] = std::move(past_key_tensor);
  inputs[4] = std::move(past_value_tensor);
  inputs[5] = std::move(seqlens_tensor);
  inputs[6] = std::move(total_seq_tensor);

  // Warm-up run
  Ort::RunOptions run_options;
  auto warmup = session.Run(run_options, input_names, inputs.data(), inputs.size(),
                            output_names, 3);

  // Re-create inputs for timed run (consumed by Run)
  inputs[0] = Ort::Value::CreateTensor(
      mem_info, query_data.data(), q_size * sizeof(uint16_t),
      q_shape.data(), q_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  inputs[1] = Ort::Value::CreateTensor(
      mem_info, key_data.data(), k_size * sizeof(uint16_t),
      k_shape.data(), k_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  inputs[2] = Ort::Value::CreateTensor(
      mem_info, value_data.data(), v_size * sizeof(uint16_t),
      v_shape.data(), v_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  inputs[3] = Ort::Value::CreateTensor(
      mem_info, past_key.data(), cache_size * sizeof(uint16_t),
      cache_shape.data(), cache_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  inputs[4] = Ort::Value::CreateTensor(
      mem_info, past_value.data(), cache_size * sizeof(uint16_t),
      cache_shape.data(), cache_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  inputs[5] = Ort::Value::CreateTensor(
      mem_info, seqlens_k_data.data(), seqlens_k_data.size() * sizeof(int32_t),
      seqlens_shape.data(), seqlens_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  inputs[6] = Ort::Value::CreateTensor(
      mem_info, total_seq_data.data(), total_seq_data.size() * sizeof(int32_t),
      total_seq_shape.data(), total_seq_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

  // Timed run (median of N iterations)
  const int NUM_ITERS = num_iters;
  std::vector<double> iter_times;
  iter_times.reserve(NUM_ITERS);
  std::vector<Ort::Value> outputs;
  for (int i = 0; i < NUM_ITERS; ++i) {
    // Re-create inputs each iteration
    inputs[0] = Ort::Value::CreateTensor(
        mem_info, query_data.data(), q_size * sizeof(uint16_t),
        q_shape.data(), q_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    inputs[1] = Ort::Value::CreateTensor(
        mem_info, key_data.data(), k_size * sizeof(uint16_t),
        k_shape.data(), k_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    inputs[2] = Ort::Value::CreateTensor(
        mem_info, value_data.data(), v_size * sizeof(uint16_t),
        v_shape.data(), v_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    inputs[3] = Ort::Value::CreateTensor(
        mem_info, past_key.data(), cache_size * sizeof(uint16_t),
        cache_shape.data(), cache_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    inputs[4] = Ort::Value::CreateTensor(
        mem_info, past_value.data(), cache_size * sizeof(uint16_t),
        cache_shape.data(), cache_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    inputs[5] = Ort::Value::CreateTensor(
        mem_info, seqlens_k_data.data(), seqlens_k_data.size() * sizeof(int32_t),
        seqlens_shape.data(), seqlens_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    inputs[6] = Ort::Value::CreateTensor(
        mem_info, total_seq_data.data(), total_seq_data.size() * sizeof(int32_t),
        total_seq_shape.data(), total_seq_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

    auto t_start = std::chrono::high_resolution_clock::now();
    outputs = session.Run(run_options, input_names, inputs.data(), inputs.size(),
                          output_names, 3);
    auto t_end = std::chrono::high_resolution_clock::now();
    iter_times.push_back(std::chrono::duration<double, std::milli>(t_end - t_start).count());
  }
  std::sort(iter_times.begin(), iter_times.end());
  double elapsed = iter_times[NUM_ITERS / 2];  // median

  // Extract results
  GqaResult result;
  result.elapsed_ms = elapsed;

  // Output
  const uint16_t* out_data = outputs[0].GetTensorData<uint16_t>();
  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  result.output.assign(out_data, out_data + out_info.GetElementCount());

  // Present key/value
  const uint16_t* pk_data = outputs[1].GetTensorData<uint16_t>();
  auto pk_info = outputs[1].GetTensorTypeAndShapeInfo();
  result.present_key.assign(pk_data, pk_data + pk_info.GetElementCount());

  const uint16_t* pv_data = outputs[2].GetTensorData<uint16_t>();
  auto pv_info = outputs[2].GetTensorTypeAndShapeInfo();
  result.present_value.assign(pv_data, pv_data + pv_info.GetElementCount());

  return result;
}

// --------------------------------------------------------------------------
// Comparison helpers
// --------------------------------------------------------------------------

// Compute max absolute difference and RMSE between two fp16 buffers
struct CompareStats {
  double max_abs_diff;
  double rmse;
  double cosine_sim;
  size_t count;
};

static CompareStats compare_fp16(const uint16_t* a, const uint16_t* b, size_t count) {
  CompareStats stats{};
  stats.count = count;
  double sum_sq_diff = 0.0;
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
  double max_diff = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double va = fp16_to_fp32(a[i]);
    double vb = fp16_to_fp32(b[i]);
    double diff = std::abs(va - vb);
    if (diff > max_diff) max_diff = diff;
    sum_sq_diff += diff * diff;
    dot += va * vb;
    norm_a += va * va;
    norm_b += vb * vb;
  }
  stats.max_abs_diff = max_diff;
  stats.rmse = std::sqrt(sum_sq_diff / count);
  double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  stats.cosine_sim = (denom > 1e-12) ? (dot / denom) : 0.0;
  return stats;
}

// Compare a token range between two BNSH buffers.
// Returns CompareStats for elements in [start_token, start_token + num_tokens).
static CompareStats compare_bnsh_token_range(const uint16_t* a, const uint16_t* b,
                                             int batch_size, int num_heads,
                                             int max_seq, int head_size,
                                             int start_token, int num_tokens) {
  size_t count = static_cast<size_t>(batch_size) * num_heads * num_tokens * head_size;
  std::vector<uint16_t> buf_a(count), buf_b(count);
  size_t idx = 0;
  for (int bx = 0; bx < batch_size; ++bx) {
    for (int h = 0; h < num_heads; ++h) {
      for (int t = 0; t < num_tokens; ++t) {
        size_t offset = (static_cast<size_t>(bx) * num_heads + h) * max_seq * head_size +
                        static_cast<size_t>(start_token + t) * head_size;
        std::memcpy(&buf_a[idx], a + offset, head_size * sizeof(uint16_t));
        std::memcpy(&buf_b[idx], b + offset, head_size * sizeof(uint16_t));
        idx += head_size;
      }
    }
  }
  return compare_fp16(buf_a.data(), buf_b.data(), count);
}

static void print_compare(const char* label, const CompareStats& s) {
  std::cout << "  " << label << ": max_diff=" << std::scientific << std::setprecision(4) << s.max_abs_diff
            << "  rmse=" << s.rmse
            << "  cosine=" << std::fixed << std::setprecision(6) << s.cosine_sim
            << "\n";
}

// --------------------------------------------------------------------------
// Create a session with WebGPU EP
// --------------------------------------------------------------------------
Ort::Session create_session(Ort::Env& env, const std::filesystem::path& model_path, bool turbo_quant) {
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  opts.DisableMemPattern();

  std::unordered_map<std::string, std::string> provider_options;
  if (turbo_quant) {
    provider_options["turboQuant"] = "1";
  }
  opts.AppendExecutionProvider("WebGPU", provider_options);
  return Ort::Session(env, model_path.native().c_str(), opts);
}

// --------------------------------------------------------------------------
// Test scenarios
// --------------------------------------------------------------------------

struct PromptResult {
  GqaResult result;
  bool valid;
  double l2_norm;
};

PromptResult run_prompt_test(Ort::Session& session, const GqaConfig& cfg,
                             int prompt_len, uint32_t seed, bool turbo_quant = false,
                             int num_iters = 3) {
  std::mt19937 rng(seed);
  auto result = run_gqa(session, cfg, prompt_len, 0, nullptr, nullptr, rng, turbo_quant, 0, num_iters);
  bool valid = check_output_valid(result.output.data(), result.output.size());
  double l2 = compute_l2_norm(result.output.data(), result.output.size());
  return {std::move(result), valid, l2};
}

struct DecodeResult {
  std::vector<GqaResult> step_results;
  double avg_latency_ms;
  double total_latency_ms;
  bool all_valid;
};

DecodeResult run_decode_test(Ort::Session& session, const GqaConfig& cfg,
                             int past_seq_len, int num_steps, uint32_t seed,
                             bool turbo_quant = false) {
  std::mt19937 rng(seed);
  const int cache_dim = cfg.cache_last_dim(turbo_quant);
  size_t cache_size = static_cast<size_t>(cfg.batch_size) * cfg.kv_num_heads * cfg.max_cache * cache_dim;

  std::vector<uint16_t> past_key(cache_size);
  std::vector<uint16_t> past_value(cache_size);
  fill_random_fp16(past_key.data(), cache_size, rng);
  fill_random_fp16(past_value.data(), cache_size, rng);

  DecodeResult dr;
  dr.all_valid = true;
  dr.total_latency_ms = 0;

  for (int step = 0; step < num_steps; ++step) {
    int current_past = past_seq_len + step;
    auto result = run_gqa(session, cfg, 1, current_past,
                          past_key.data(), past_value.data(), rng, turbo_quant);
    bool valid = check_output_valid(result.output.data(), result.output.size());
    if (!valid) dr.all_valid = false;
    dr.total_latency_ms += result.elapsed_ms;
    past_key = result.present_key;
    past_value = result.present_value;
    dr.step_results.push_back(std::move(result));
  }
  dr.avg_latency_ms = dr.total_latency_ms / num_steps;
  return dr;
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gqa_op_test");
    std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << "\n";

    const std::filesystem::path model_path = (argc > 1) ? argv[1] : "gqa_model.onnx";
    std::cout << "Model: " << model_path.string() << "\n\n";

    // ----- Create both sessions -----
    std::cout << "Creating baseline session (no TurboQuant)...\n";
    auto session_base = create_session(env, model_path, false);

    std::cout << "Creating TurboQuant session...\n";
    auto session_tq = create_session(env, model_path, true);

    // Print model info (once)
    Ort::AllocatorWithDefaultOptions allocator;
    std::cout << "\nModel inputs:  " << session_base.GetInputCount() << "\n";
    for (size_t i = 0; i < session_base.GetInputCount(); ++i) {
      auto name = session_base.GetInputNameAllocated(i, allocator);
      auto info = session_base.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();
      std::cout << "  [" << i << "] " << name.get() << " shape=[";
      for (size_t j = 0; j < shape.size(); ++j) {
        std::cout << (j ? "," : "") << shape[j];
      }
      std::cout << "]\n";
    }
    std::cout << "Model outputs: " << session_base.GetOutputCount() << "\n\n";

    GqaConfig cfg;
    cfg.batch_size = 1;
    cfg.num_heads = 24;
    cfg.kv_num_heads = 8;
    cfg.head_size = 128;
    cfg.max_cache = 131072;
    cfg.init_compressed_dim();

    // =====================================================================
    //  Prompt Tests — Correctness & Performance Comparison
    // =====================================================================
    std::cout << "============================================================\n";
    std::cout << " Prompt Tests: Baseline vs TurboQuant\n";
    std::cout << "============================================================\n";

    const int prompt_lengths[] = {1, 8, 64, 256, 1024};
    for (int plen : prompt_lengths) {
      uint32_t seed = 10000 + plen;
      std::cout << "\n--- seq_len=" << plen << " ---\n";

      auto base = run_prompt_test(session_base, cfg, plen, seed, false);
      auto tq = run_prompt_test(session_tq, cfg, plen, seed, true);

      std::cout << "  Baseline:    L2=" << std::fixed << std::setprecision(4) << base.l2_norm
                << "  latency=" << std::setprecision(2) << base.result.elapsed_ms << " ms"
                << "  valid=" << (base.valid ? "Y" : "N") << "\n";
      std::cout << "  TurboQuant:  L2=" << std::fixed << std::setprecision(4) << tq.l2_norm
                << "  latency=" << std::setprecision(2) << tq.result.elapsed_ms << " ms"
                << "  valid=" << (tq.valid ? "Y" : "N") << "\n";

      THROW_IF_NOT(base.valid);
      THROW_IF_NOT(tq.valid);

      // Compare outputs — quantization introduces error, so relax threshold
      auto cmp = compare_fp16(base.result.output.data(), tq.result.output.data(),
                              base.result.output.size());
      print_compare("Output diff", cmp);

      bool close = (cmp.cosine_sim > 0.90);
      std::cout << "  Match:       " << (close ? "PASS" : "FAIL")
                << " (cosine > 0.90)\n";
      THROW_IF_NOT(close);

      // Verify present_value: rotate+quantize+pack baseline's V on CPU and compare packed indices with GPU.
      // For prompt (past_seq=0), ALL tokens [0..plen) are newly written.
      {
        size_t compressed_cache_size = static_cast<size_t>(cfg.batch_size) * cfg.kv_num_heads * cfg.max_cache * cfg.compressed_dim;
        std::vector<uint16_t> cpu_pv_packed(compressed_cache_size, 0);
        cpu_rotate_quantize_packed_bnsh(base.result.present_value.data(), cpu_pv_packed.data(),
                                        cfg.batch_size, cfg.kv_num_heads,
                                        cfg.max_cache, cfg.head_size, cfg.compressed_dim, 0, plen);
        double idx_match = compare_packed_indices(
            cpu_pv_packed.data(), tq.result.present_value.data(),
            cfg.batch_size, cfg.kv_num_heads, cfg.max_cache, cfg.head_size, cfg.compressed_dim,
            0, plen);
        std::cout << "  V quant index match: " << std::fixed << std::setprecision(4)
                  << (idx_match * 100.0) << "%\n";
        THROW_IF_NOT(idx_match > 0.95);
      }
    }

    // =====================================================================
    //  Decode Tests — Non-shared buffer (dynamic KV cache)
    //  The EP copies compressed past to new present, then quantizes new tokens.
    //  Step 1: Run a prompt to get compressed present (step0)
    //  Step 2: Feed compressed present back as past for a decode step
    // =====================================================================
    std::cout << "\n============================================================\n";
    std::cout << " Decode Tests: Non-shared buffer (dynamic KV cache)\n";
    std::cout << "============================================================\n";
    std::cout << " Flow: prompt -> compressed present -> feed back as past -> decode\n";
    std::cout << "============================================================\n";

    const int prompt_lens[] = {1, 8, 64, 256};

    for (int plen : prompt_lens) {
      uint32_t seed = 30000 + plen;
      std::cout << "\n--- prompt=" << plen << " + 1 decode step ---\n";

      // Run identical prompt on both sessions
      std::mt19937 rng_b1(seed), rng_t1(seed);
      auto base_prompt = run_gqa(session_base, cfg, plen, 0,
                                 nullptr, nullptr, rng_b1);
      auto tq_prompt = run_gqa(session_tq, cfg, plen, 0,
                                nullptr, nullptr, rng_t1, true);

      // Now run a decode step: feed back present as past
      // Baseline: present_key/value have shape [B,H,max,128] (uncompressed)
      // TQ:       present_key/value have shape [B,H,max,36] (compressed)
      uint32_t decode_seed = seed + 50000;
      std::mt19937 rng_b2(decode_seed), rng_t2(decode_seed);

      auto base_decode = run_gqa(session_base, cfg, 1, plen,
                                 base_prompt.present_key.data(),
                                 base_prompt.present_value.data(),
                                 rng_b2);
      auto tq_decode = run_gqa(session_tq, cfg, 1, plen,
                                tq_prompt.present_key.data(),
                                tq_prompt.present_value.data(),
                                rng_t2, true,
                                cfg.compressed_dim);  // past is compressed

      bool bv = check_output_valid(base_decode.output.data(), base_decode.output.size());
      bool tv = check_output_valid(tq_decode.output.data(), tq_decode.output.size());

      auto out_cmp = compare_fp16(base_decode.output.data(), tq_decode.output.data(),
                                  base_decode.output.size());

      // Compare the NEW token's V quantization (at position plen)
      size_t cpu_pv_size = static_cast<size_t>(cfg.batch_size) * cfg.kv_num_heads * cfg.max_cache * cfg.compressed_dim;
      std::vector<uint16_t> cpu_pv_packed(cpu_pv_size, 0);
      cpu_rotate_quantize_packed_bnsh(base_decode.present_value.data(), cpu_pv_packed.data(),
                                      cfg.batch_size, cfg.kv_num_heads,
                                      cfg.max_cache, cfg.head_size, cfg.compressed_dim,
                                      plen, 1);
      double pv_idx_match = compare_packed_indices(
          cpu_pv_packed.data(), tq_decode.present_value.data(),
          cfg.batch_size, cfg.kv_num_heads, cfg.max_cache, cfg.head_size, cfg.compressed_dim,
          plen, 1);

      std::cout << "  valid=" << (bv && tv ? "Y" : "N")
                << "  output_cos=" << std::fixed << std::setprecision(6) << out_cmp.cosine_sim
                << "  output_maxdiff=" << std::scientific << std::setprecision(4) << out_cmp.max_abs_diff
                << "  V_idx=" << std::fixed << std::setprecision(1) << (pv_idx_match * 100.0) << "%"
                << "\n";
      std::cout << "  latency: base=" << std::fixed << std::setprecision(2) << base_decode.elapsed_ms
                << " ms  tq=" << tq_decode.elapsed_ms << " ms\n";

      THROW_IF_NOT(bv && tv);
      THROW_IF_NOT(pv_idx_match > 0.95);
      THROW_IF_NOT(out_cmp.cosine_sim > 0.85);
      std::cout << "  Result: PASS\n";
    }

    // =====================================================================
    //  Multi-step Decode Test — Chain prompt + N decode steps
    //  Exercises the full flow: prompt → present(compressed) → past → decode → ...
    // =====================================================================
    std::cout << "\n============================================================\n";
    std::cout << " Multi-step Decode: prompt(8) + 5 decode steps\n";
    std::cout << "============================================================\n";

    {
      const int initial_prompt = 8;
      const int num_decode_steps = 5;
      uint32_t seed = 40000;

      // Prompt step — identical for both
      std::mt19937 rng_b(seed), rng_t(seed);
      auto base_prev = run_gqa(session_base, cfg, initial_prompt, 0,
                                nullptr, nullptr, rng_b);
      auto tq_prev = run_gqa(session_tq, cfg, initial_prompt, 0,
                              nullptr, nullptr, rng_t, true);

      for (int step = 0; step < num_decode_steps; ++step) {
        int past_len = initial_prompt + step;
        uint32_t step_seed = seed + 1000 * (step + 1);
        std::mt19937 rng_bs(step_seed), rng_ts(step_seed);

        auto base_cur = run_gqa(session_base, cfg, 1, past_len,
                                 base_prev.present_key.data(),
                                 base_prev.present_value.data(),
                                 rng_bs);

        // TQ past is compressed (dim=36)
        int tq_past_dim = (step == 0) ? cfg.compressed_dim : cfg.compressed_dim;
        auto tq_cur = run_gqa(session_tq, cfg, 1, past_len,
                               tq_prev.present_key.data(),
                               tq_prev.present_value.data(),
                               rng_ts, true, tq_past_dim);

        bool bv = check_output_valid(base_cur.output.data(), base_cur.output.size());
        bool tv = check_output_valid(tq_cur.output.data(), tq_cur.output.size());
        auto out_cmp = compare_fp16(base_cur.output.data(), tq_cur.output.data(),
                                    base_cur.output.size());

        std::cout << "  step " << (step + 1)
                  << ": valid=" << (bv && tv ? "Y" : "N")
                  << "  cos=" << std::fixed << std::setprecision(6) << out_cmp.cosine_sim
                  << "  maxdiff=" << std::scientific << std::setprecision(4) << out_cmp.max_abs_diff
                  << "\n";

        THROW_IF_NOT(bv && tv);
        THROW_IF_NOT(out_cmp.cosine_sim > 0.85);

        base_prev = std::move(base_cur);
        tq_prev = std::move(tq_cur);
      }
      std::cout << "  Result: PASS (all " << num_decode_steps << " steps)\n";
    }

    // =====================================================================
    //  Consistency Test
    // =====================================================================
    std::cout << "\n============================================================\n";
    std::cout << " Consistency: same input -> same output (both modes)\n";
    std::cout << "============================================================\n";

    {
      auto r1 = run_prompt_test(session_base, cfg, 8, 42);
      auto r2 = run_prompt_test(session_base, cfg, 8, 42);
      bool match = std::memcmp(r1.result.output.data(), r2.result.output.data(),
                               r1.result.output.size() * sizeof(uint16_t)) == 0;
      std::cout << "  Baseline consistency: " << (match ? "PASS" : "FAIL") << "\n";
      THROW_IF_NOT(match);
    }
    {
      auto r1 = run_prompt_test(session_tq, cfg, 8, 42, true);
      auto r2 = run_prompt_test(session_tq, cfg, 8, 42, true);
      bool match = std::memcmp(r1.result.output.data(), r2.result.output.data(),
                               r1.result.output.size() * sizeof(uint16_t)) == 0;
      std::cout << "  TurboQuant consistency: " << (match ? "PASS" : "FAIL") << "\n";
      THROW_IF_NOT(match);
    }

    // =====================================================================
    //  Quality vs Sequence Length — How does TQ accuracy scale?
    // =====================================================================
    std::cout << "\n============================================================\n";
    std::cout << " Quality vs Sequence Length (single iteration each)\n";
    std::cout << "============================================================\n";
    std::cout << "  " << std::left << std::setw(12) << "seq_len"
              << std::setw(12) << "cosine"
              << std::setw(14) << "max_diff"
              << std::setw(12) << "rmse"
              << std::setw(16) << "base_ms"
              << std::setw(16) << "tq_ms"
              << std::setw(10) << "speedup" << "\n";
    std::cout << "  " << std::string(92, '-') << "\n";

    for (int plen : {1, 8, 64, 256, 1024, 4096, 16384, 32768, 131072}) {
      if (plen > cfg.max_cache) break;  // skip if model can't handle it
      uint32_t seed = 50000 + plen;

      // Use a right-sized config for this test to avoid OOM from oversized cache buffers.
      // The model's past dim[2] is dynamic (-1), so any cache size works.
      GqaConfig qcfg = cfg;
      qcfg.max_cache = plen;  // cache only needs to hold the prompt tokens

      auto base = run_prompt_test(session_base, qcfg, plen, seed, false, 1);
      auto tq = run_prompt_test(session_tq, qcfg, plen, seed, true, 1);
      auto cmp = compare_fp16(base.result.output.data(), tq.result.output.data(),
                              base.result.output.size());
      double speedup = base.result.elapsed_ms / tq.result.elapsed_ms;
      std::cout << "  " << std::left << std::setw(12) << plen
                << std::fixed << std::setprecision(6) << std::setw(12) << cmp.cosine_sim
                << std::scientific << std::setprecision(4) << std::setw(14) << cmp.max_abs_diff
                << std::setw(12) << cmp.rmse
                << std::fixed << std::setprecision(2) << std::setw(16) << base.result.elapsed_ms
                << std::setw(16) << tq.result.elapsed_ms
                << std::setw(10) << speedup << "x\n";
      std::cout.flush();
    }

    // =====================================================================
    //  Performance Summary
    // =====================================================================
    std::cout << "\n============================================================\n";
    std::cout << " Performance Summary\n";
    std::cout << "============================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::left << std::setw(20) << "Test"
              << std::setw(15) << "Baseline (ms)"
              << std::setw(15) << "TurboQ (ms)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << "  " << std::string(62, '-') << "\n";

    // Re-run key benchmarks for summary table
    for (int plen : {1, 64, 256, 1024}) {
      uint32_t seed = 30000 + plen;
      auto base = run_prompt_test(session_base, cfg, plen, seed, false);
      auto tq = run_prompt_test(session_tq, cfg, plen, seed, true);
      double speedup = base.result.elapsed_ms / tq.result.elapsed_ms;
      std::string label = "prompt_" + std::to_string(plen);
      std::cout << "  " << std::left << std::setw(20) << label
                << std::setw(15) << base.result.elapsed_ms
                << std::setw(15) << tq.result.elapsed_ms
                << std::setw(12) << speedup << "x\n";
    }
    for (int depth : {256, 1024, 4000}) {
      uint32_t seed = 40000 + depth;
      auto base = run_decode_test(session_base, cfg, depth, 3, seed, false);
      auto tq = run_decode_test(session_tq, cfg, depth, 3, seed, true);
      double speedup = base.avg_latency_ms / tq.avg_latency_ms;
      std::string label = "decode@" + std::to_string(depth);
      std::cout << "  " << std::left << std::setw(20) << label
                << std::setw(15) << base.avg_latency_ms
                << std::setw(15) << tq.avg_latency_ms
                << std::setw(12) << speedup << "x\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << " ALL TESTS PASSED\n";
    std::cout << "============================================================\n";

  } catch (const std::exception& e) {
    std::cerr << "\nFATAL: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
