// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Operator-level correctness tests for PagedAttention. Each test compares the
// provider output and updated caches against a CPU scaled-dot-product-attention
// reference, and runs against each provider available in the build.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "default_providers.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

namespace {

// Index into a (num_blocks, block_size, kv_num_heads, head_size) tensor.
int CacheIndex(int block_id, int slot_in_block, int kv_head, int dim,
               int block_size, int kv_num_heads, int head_size) {
  return ((block_id * block_size + slot_in_block) * kv_num_heads + kv_head) * head_size + dim;
}

}  // namespace

namespace {

struct EndToEndCase {
  int batch_size = 1;
  int token_count = 1;
  int num_heads = 1;
  int kv_num_heads = 1;
  int head_size = 8;
  int block_size = 256;
  int num_blocks = 2;
  int max_num_blocks_per_seq = 1;
  float scale = 0.0f;  // 0 means kernel default = 1/sqrt(head_size)
  std::vector<int32_t> cumulative_seqlens_q;
  std::vector<int32_t> past_seqlens;
  std::vector<int32_t> block_table;
};

struct IoBindingCase {
  int batch_size = 1;
  int num_heads = 1;
  int kv_num_heads = 1;
  int head_size = 8;
  int block_size = 256;
  int num_blocks = 2;
  int max_num_blocks_per_seq = 1;
  int past_seqlen = 4;
  bool split_sensitive_values = false;
  bool int8_cache = false;
  bool fp8_cache = false;
  bool bf16_query = false;
  bool enable_cuda_graph = false;
  bool irregular_layout = false;
  bool discriminating_attention = false;
  std::vector<std::vector<int32_t>> replay_past_seqlens;
  std::vector<int32_t> block_table;
  std::vector<int32_t> attention_metadata;
  std::string expected_error;
  std::string expected_initialization_error;
};

// Softmax with causal masking: masked positions get -inf → 0 after exp.
// Uses fp32 throughout to establish a reference the fp16 kernel is compared
// against with a loose tolerance.
void CausalSoftmax(std::vector<float>& scores, int q_pos) {
  const int len = static_cast<int>(scores.size());
  float max_val = -std::numeric_limits<float>::infinity();
  for (int i = 0; i <= q_pos && i < len; ++i) {
    max_val = std::max(max_val, scores[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < len; ++i) {
    if (i > q_pos) {
      scores[i] = 0.0f;
    } else {
      scores[i] = std::exp(scores[i] - max_val);
      sum += scores[i];
    }
  }
  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < len; ++i) {
    scores[i] *= inv_sum;
  }
}

void RunEndToEndCase(const EndToEndCase& c, std::unique_ptr<IExecutionProvider> execution_provider) {
  ASSERT_EQ(c.cumulative_seqlens_q.size(), static_cast<size_t>(c.batch_size + 1));
  ASSERT_EQ(c.past_seqlens.size(), static_cast<size_t>(c.batch_size));
  ASSERT_EQ(c.block_table.size(), static_cast<size_t>(c.batch_size * c.max_num_blocks_per_seq));
  ASSERT_EQ(c.cumulative_seqlens_q.back(), c.token_count);
  ASSERT_EQ(c.num_heads % c.kv_num_heads, 0);

  const int hidden_size = c.num_heads * c.head_size;
  const int kv_hidden_size = c.kv_num_heads * c.head_size;
  const int cache_elems = c.num_blocks * c.block_size * c.kv_num_heads * c.head_size;
  const int gqa_factor = c.num_heads / c.kv_num_heads;

  // ----- Inputs: tight values so fp16 SDPA stays within tolerance.
  std::vector<float> query_f(c.token_count * hidden_size);
  std::vector<float> key_f(c.token_count * kv_hidden_size);
  std::vector<float> value_f(c.token_count * kv_hidden_size);
  for (size_t i = 0; i < query_f.size(); ++i) {
    // Small range so QK products stay ~O(head_size * 0.01) → well within fp16.
    query_f[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 7) - 3);
  }
  for (size_t i = 0; i < key_f.size(); ++i) {
    key_f[i] = 0.03f * static_cast<float>((static_cast<int>(i) % 5) - 2);
    value_f[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 11) - 5);
  }
  std::vector<float> key_cache_f(cache_elems);
  std::vector<float> value_cache_f(cache_elems);
  for (int i = 0; i < cache_elems; ++i) {
    key_cache_f[i] = 0.01f * static_cast<float>((i % 9) - 4);
    value_cache_f[i] = 0.02f * static_cast<float>((i % 13) - 6);
  }

  const float scale = c.scale != 0.0f ? c.scale : 1.0f / std::sqrt(static_cast<float>(c.head_size));

  // ----- CPU reference.
  // 1) Apply scatter to a copy of the initial cache. This gives us the K/V
  //    that the PagedAttention kernel will see for each batch.
  std::vector<float> expected_key_cache_f = key_cache_f;
  std::vector<float> expected_value_cache_f = value_cache_f;
  for (int t = 0; t < c.token_count; ++t) {
    int seq_idx = 0;
    for (int b = 0; b < c.batch_size; ++b) {
      if (t < c.cumulative_seqlens_q[b + 1]) {
        seq_idx = b;
        break;
      }
    }
    const int local_tok = t - c.cumulative_seqlens_q[seq_idx];
    const int abs_slot = c.past_seqlens[seq_idx] + local_tok;
    const int block_off = abs_slot / c.block_size;
    const int slot_in_bk = abs_slot % c.block_size;
    const int block_id = c.block_table[seq_idx * c.max_num_blocks_per_seq + block_off];
    for (int h = 0; h < c.kv_num_heads; ++h) {
      for (int d = 0; d < c.head_size; ++d) {
        const int src = t * kv_hidden_size + h * c.head_size + d;
        const int dst = CacheIndex(block_id, slot_in_bk, h, d, c.block_size, c.kv_num_heads, c.head_size);
        expected_key_cache_f[dst] = key_f[src];
        expected_value_cache_f[dst] = value_f[src];
      }
    }
  }

  // 2) For each token, run causal SDPA against the (past + new) K/V window.
  std::vector<float> expected_output_f(c.token_count * hidden_size, 0.0f);
  for (int b = 0; b < c.batch_size; ++b) {
    const int cum_lo = c.cumulative_seqlens_q[b];
    const int cum_hi = c.cumulative_seqlens_q[b + 1];
    const int q_len = cum_hi - cum_lo;
    const int past = c.past_seqlens[b];
    const int total_kv_len = past + q_len;

    // Build K/V for this batch: shape (kv_num_heads, total_kv_len, head_size).
    // Each slot reads from the (post-scatter) paged cache.
    std::vector<float> k_window(c.kv_num_heads * total_kv_len * c.head_size);
    std::vector<float> v_window(c.kv_num_heads * total_kv_len * c.head_size);
    for (int s = 0; s < total_kv_len; ++s) {
      const int abs_slot = s;
      const int block_off = abs_slot / c.block_size;
      const int slot_in_bk = abs_slot % c.block_size;
      const int block_id = c.block_table[b * c.max_num_blocks_per_seq + block_off];
      for (int h = 0; h < c.kv_num_heads; ++h) {
        for (int d = 0; d < c.head_size; ++d) {
          const int src = CacheIndex(block_id, slot_in_bk, h, d, c.block_size, c.kv_num_heads, c.head_size);
          const int dst = (h * total_kv_len + s) * c.head_size + d;
          k_window[dst] = expected_key_cache_f[src];
          v_window[dst] = expected_value_cache_f[src];
        }
      }
    }

    // For each new query token, do causal SDPA.
    for (int local_tok = 0; local_tok < q_len; ++local_tok) {
      const int t = cum_lo + local_tok;
      const int q_pos = past + local_tok;  // Absolute position in the KV window.
      for (int n_q = 0; n_q < c.num_heads; ++n_q) {
        const int h_kv = n_q / gqa_factor;

        // scores[s] = scale * dot(q, k[s])
        std::vector<float> scores(total_kv_len, 0.0f);
        for (int s = 0; s < total_kv_len; ++s) {
          float dot = 0.0f;
          for (int d = 0; d < c.head_size; ++d) {
            const float q = query_f[t * hidden_size + n_q * c.head_size + d];
            const float k = k_window[(h_kv * total_kv_len + s) * c.head_size + d];
            dot += q * k;
          }
          scores[s] = dot * scale;
        }
        CausalSoftmax(scores, q_pos);

        // out[t, n_q * head_size + d] = sum_s scores[s] * v[s, d]
        for (int d = 0; d < c.head_size; ++d) {
          float acc = 0.0f;
          for (int s = 0; s <= q_pos && s < total_kv_len; ++s) {
            const float v = v_window[(h_kv * total_kv_len + s) * c.head_size + d];
            acc += scores[s] * v;
          }
          expected_output_f[t * hidden_size + n_q * c.head_size + d] = acc;
        }
      }
    }
  }

  OpTester test("PagedAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", c.num_heads);
  test.AddAttribute<int64_t>("kv_num_heads", c.kv_num_heads);
  test.AddAttribute<float>("scale", c.scale);
  test.AddAttribute<int64_t>("do_rotary", 0);

  test.AddInput<MLFloat16>("query", {c.token_count, hidden_size}, FloatsToMLFloat16s(query_f));
  test.AddInput<MLFloat16>("key", {c.token_count, kv_hidden_size}, FloatsToMLFloat16s(key_f));
  test.AddInput<MLFloat16>("value", {c.token_count, kv_hidden_size}, FloatsToMLFloat16s(value_f));
  test.AddInput<MLFloat16>("key_cache",
                           {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                           FloatsToMLFloat16s(key_cache_f));
  test.AddInput<MLFloat16>("value_cache",
                           {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                           FloatsToMLFloat16s(value_cache_f));
  test.AddInput<int32_t>("cumulative_sequence_length", {c.batch_size + 1}, c.cumulative_seqlens_q);
  test.AddInput<int32_t>("past_seqlens", {c.batch_size}, c.past_seqlens);
  test.AddInput<int32_t>("block_table", {c.batch_size, c.max_num_blocks_per_seq}, c.block_table);
  test.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  test.AddOptionalInputEdge<MLFloat16>();  // sin_cache

  test.AddOutput<MLFloat16>("output", {c.token_count, hidden_size}, FloatsToMLFloat16s(expected_output_f));
  test.AddOutput<MLFloat16>("key_cache_out",
                            {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                            FloatsToMLFloat16s(expected_key_cache_f));
  test.AddOutput<MLFloat16>("value_cache_out",
                            {c.num_blocks, c.block_size, c.kv_num_heads, c.head_size},
                            FloatsToMLFloat16s(expected_value_cache_f));

  // fp16 softmax + dot-product accumulation over a short KV window.
  // Reference is fp32; kernel is fp16. 2e-2 is a comfortable envelope for
  // these value ranges and short sequences.
  test.SetOutputAbsErr("output", 2e-2f);
  test.SetOutputAbsErr("key_cache_out", 1e-3f);
  test.SetOutputAbsErr("value_cache_out", 1e-3f);

  test.ConfigEp(std::move(execution_provider)).RunWithConfig();
}

void RunIoBindingCase(std::unique_ptr<IExecutionProvider> execution_provider,
                      const char* provider_type,
                      bool alias_cache_outputs,
                      bool omit_cache_outputs = false,
                      const IoBindingCase& c = IoBindingCase{}) {
  const int batch_size = c.batch_size;
  const int token_count = batch_size;
  const int num_heads = c.num_heads;
  const int kv_num_heads = c.kv_num_heads;
  const int head_size = c.head_size;
  const int block_size = c.block_size;
  const int num_blocks = c.num_blocks;
  const int max_num_blocks_per_seq = c.max_num_blocks_per_seq;
  const int past_seqlen = c.past_seqlen;
  const int hidden_size = num_heads * head_size;
  const int kv_hidden_size = kv_num_heads * head_size;
  const int cache_elems = num_blocks * block_size * kv_num_heads * head_size;
  constexpr float cache_scale = 0.01f;
  const bool quantized_cache = c.int8_cache || c.fp8_cache;

  ASSERT_FALSE(c.int8_cache && c.fp8_cache);
#if defined(DISABLE_FLOAT8_TYPES)
  ASSERT_FALSE(c.fp8_cache);
#endif
  ASSERT_FALSE(c.replay_past_seqlens.empty() && c.enable_cuda_graph);
  for (const auto& replay_lengths : c.replay_past_seqlens) {
    ASSERT_EQ(replay_lengths.size(), static_cast<size_t>(batch_size));
    for (int32_t replay_length : replay_lengths) {
      ASSERT_GE(replay_length, 0);
      ASSERT_GT(max_num_blocks_per_seq, replay_length / block_size);
    }
  }
  ASSERT_TRUE(!c.replay_past_seqlens.empty() || max_num_blocks_per_seq > past_seqlen / block_size);
  ASSERT_LE(batch_size * max_num_blocks_per_seq, num_blocks);
  ASSERT_EQ(num_heads % kv_num_heads, 0);
  ASSERT_TRUE(c.block_table.empty() ||
              c.block_table.size() == static_cast<size_t>(batch_size * max_num_blocks_per_seq));
  for (int32_t block_id : c.block_table) {
    ASSERT_GE(block_id, 0);
    ASSERT_LT(block_id, num_blocks);
  }

  std::unordered_map<std::string, int> domain_to_version = {{onnxruntime::kMSDomain, 1}};
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  auto model = std::make_unique<Model>("paged_attention_cuda_alias_test", true, ModelMetaData(), PathString(),
                                       IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                                       model_specific_functions, DefaultLoggingManager().DefaultLogger(),
                                       ModelOptions(true, true));
  auto& graph = model->MainGraph();

  std::vector<ONNX_NAMESPACE::TypeProto> tensor_types;
  tensor_types.reserve(13);
  auto add_tensor_type = [&](int elem_type, std::initializer_list<int64_t> dims) -> ONNX_NAMESPACE::TypeProto* {
    tensor_types.emplace_back();
    auto* type = &tensor_types.back();
    type->mutable_tensor_type()->set_elem_type(elem_type);
    auto* shape = type->mutable_tensor_type()->mutable_shape();
    for (const int64_t dim : dims) {
      shape->add_dim()->set_dim_value(dim);
    }
    return type;
  };

  const int query_elem_type = c.bf16_query ? ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16
                                           : ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  auto& query_arg = graph.GetOrCreateNodeArg("query", add_tensor_type(query_elem_type,
                                                                      {token_count, hidden_size}));
  auto& key_arg = graph.GetOrCreateNodeArg("key", add_tensor_type(query_elem_type,
                                                                  {token_count, kv_hidden_size}));
  auto& value_arg = graph.GetOrCreateNodeArg("value", add_tensor_type(query_elem_type,
                                                                      {token_count, kv_hidden_size}));
  const int cache_elem_type = c.int8_cache
                                  ? ONNX_NAMESPACE::TensorProto_DataType_INT8
                              : c.fp8_cache
                                  ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN
                                  : query_elem_type;
  auto& key_cache_arg = graph.GetOrCreateNodeArg(
      "key_cache", add_tensor_type(cache_elem_type,
                                   {num_blocks, block_size, kv_num_heads, head_size}));
  auto& value_cache_arg = graph.GetOrCreateNodeArg(
      "value_cache", add_tensor_type(cache_elem_type,
                                     {num_blocks, block_size, kv_num_heads, head_size}));
  auto& cumulative_sequence_length_arg = graph.GetOrCreateNodeArg(
      "cumulative_sequence_length", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch_size + 1}));
  auto& past_seqlens_arg = graph.GetOrCreateNodeArg(
      "past_seqlens", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch_size}));
  auto& block_table_arg = graph.GetOrCreateNodeArg(
      "block_table",
      add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch_size, max_num_blocks_per_seq}));
  auto& empty_optional_arg = graph.GetOrCreateNodeArg("", nullptr);
  NodeArg* attention_metadata_arg = &empty_optional_arg;
  if (!c.attention_metadata.empty()) {
    attention_metadata_arg = &graph.GetOrCreateNodeArg(
        "attention_metadata",
        add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32,
                        {static_cast<int64_t>(c.attention_metadata.size())}));
  }
  NodeArg* k_scale_arg = &empty_optional_arg;
  NodeArg* v_scale_arg = &empty_optional_arg;
  if (quantized_cache) {
    k_scale_arg = &graph.GetOrCreateNodeArg(
        "k_scale", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1}));
    v_scale_arg = &graph.GetOrCreateNodeArg(
        "v_scale", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1}));
  }
  std::vector<NodeArg*> input_defs = {&query_arg, &key_arg, &value_arg, &key_cache_arg, &value_cache_arg,
                                      &cumulative_sequence_length_arg, &past_seqlens_arg, &block_table_arg,
                                      /*cos_cache=*/&empty_optional_arg,
                                      /*sin_cache=*/&empty_optional_arg,
                                      /*slot_mapping=*/&empty_optional_arg,
                                      /*head_sink=*/&empty_optional_arg,
                                      /*q_norm_weight=*/&empty_optional_arg,
                                      /*k_norm_weight=*/&empty_optional_arg,
                                      k_scale_arg,
                                      v_scale_arg,
                                      attention_metadata_arg};

  auto& output_arg = graph.GetOrCreateNodeArg(
      "output", add_tensor_type(query_elem_type, {token_count, hidden_size}));
  auto& key_cache_out_arg = graph.GetOrCreateNodeArg(
      "key_cache_out", add_tensor_type(cache_elem_type,
                                       {num_blocks, block_size, kv_num_heads, head_size}));
  auto& value_cache_out_arg = graph.GetOrCreateNodeArg(
      "value_cache_out", add_tensor_type(cache_elem_type,
                                         {num_blocks, block_size, kv_num_heads, head_size}));
  std::vector<NodeArg*> output_defs = {&output_arg};
  if (!omit_cache_outputs) {
    output_defs.push_back(&key_cache_out_arg);
    output_defs.push_back(&value_cache_out_arg);
  }

  NodeAttributes attrs = {
      {"num_heads", utils::MakeAttribute("num_heads", int64_t{num_heads})},
      {"kv_num_heads", utils::MakeAttribute("kv_num_heads", int64_t{kv_num_heads})},
      {"scale", utils::MakeAttribute("scale", 0.0f)},
      {"do_rotary", utils::MakeAttribute("do_rotary", int64_t{0})},
  };
  if (quantized_cache) {
    attrs.emplace("k_quant_type", utils::MakeAttribute("k_quant_type", std::string{"PER_TENSOR"}));
    attrs.emplace("v_quant_type", utils::MakeAttribute("v_quant_type", std::string{"PER_TENSOR"}));
  }
  auto& node = graph.AddNode("paged_attention", "PagedAttention", "IOBinding cache test",
                             input_defs, output_defs, &attrs, onnxruntime::kMSDomain);
  node.SetExecutionProviderType(provider_type);
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_string;
  ASSERT_TRUE(model->ToProto().SerializeToString(&model_string));
  std::stringstream model_stream(model_string);

  SessionOptions session_options;
  session_options.session_logid = "PagedAttentionIoBindingTest";
  InferenceSession session(session_options, GetEnvironment());
  ASSERT_NE(execution_provider, nullptr);
  IExecutionProvider* execution_provider_ptr = execution_provider.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(execution_provider)));
  auto device_allocators = execution_provider_ptr->CreatePreferredAllocators();
  ASSERT_FALSE(device_allocators.empty());
  const OrtMemoryInfo* selected_device_memory_info = nullptr;
  for (const auto& allocator : device_allocators) {
    const auto& mem_info = allocator->Info();
    if (mem_info.device.Type() == OrtDevice::GPU && mem_info.mem_type == OrtMemTypeDefault) {
      selected_device_memory_info = &mem_info;
    }
  }
  ASSERT_NE(selected_device_memory_info, nullptr);
  const OrtMemoryInfo device_memory_info = *selected_device_memory_info;
  ASSERT_STATUS_OK(session.Load(model_stream));
  const Status initialization_status = session.Initialize();
  if (!c.expected_initialization_error.empty()) {
    EXPECT_FALSE(initialization_status.IsOK());
    EXPECT_THAT(initialization_status.ErrorMessage(), testing::HasSubstr(c.expected_initialization_error));
    return;
  }
  ASSERT_STATUS_OK(initialization_status);
  auto device_alloc = session.GetAllocator(device_memory_info);
  ASSERT_NE(device_alloc, nullptr);

  auto cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  OrtValue attention_metadata_value;
  if (!c.attention_metadata.empty()) {
    Tensor cpu_tensor(DataTypeImpl::GetType<int32_t>(),
                      TensorShape({static_cast<int64_t>(c.attention_metadata.size())}),
                      const_cast<int32_t*>(c.attention_metadata.data()), cpu_alloc->Info());
    Tensor::InitOrtValue(std::move(cpu_tensor), attention_metadata_value);
  }

  auto make_gpu = [&](const auto& data, const TensorShape& shape) {
    using T = typename std::decay_t<decltype(data)>::value_type;
    Tensor cpu_tensor(DataTypeImpl::GetType<T>(), shape, const_cast<T*>(data.data()), cpu_alloc->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<T>(), shape, device_alloc);
    ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue value;
    Tensor::InitOrtValue(std::move(gpu_tensor), value);
    return value;
  };

  std::vector<int32_t> block_table_data = c.block_table;
  if (block_table_data.empty()) {
    block_table_data.resize(batch_size * max_num_blocks_per_seq);
    std::iota(block_table_data.begin(), block_table_data.end(), 0);
  }

  std::vector<MLFloat16> query_data(token_count * hidden_size, MLFloat16(0.02f));
  std::vector<MLFloat16> key_data(token_count * kv_hidden_size, MLFloat16(0.03f));
  std::vector<MLFloat16> value_data(token_count * kv_hidden_size);
  for (int b = 0; b < batch_size; ++b) {
    const MLFloat16 value(0.04f + 0.02f * b);
    std::fill_n(value_data.begin() + b * kv_hidden_size, kv_hidden_size, value);
  }
  std::vector<MLFloat16> key_cache_data(cache_elems, MLFloat16(0.01f));
  std::vector<MLFloat16> value_cache_data(cache_elems, MLFloat16(0.02f));
  if (c.irregular_layout) {
    const auto walsh_sign = [](int pattern, int dim) {
      return std::popcount(static_cast<unsigned>(pattern & (dim % 8))) % 2 == 0 ? 1.0f : -1.0f;
    };
    const float key_step = quantized_cache ? cache_scale : 0.003f;
    const float value_step = quantized_cache ? cache_scale : 0.004f;
    const float cache_key_step = quantized_cache ? cache_scale : 0.001f;
    const float cache_value_step = quantized_cache ? cache_scale : 0.003f;
    for (int b = 0; b < batch_size; ++b) {
      for (int q_head = 0; q_head < num_heads; ++q_head) {
        for (int dim = 0; dim < head_size; ++dim) {
          const int index = (b * num_heads + q_head) * head_size + dim;
          query_data[index] = c.discriminating_attention
                                  ? MLFloat16(0.5f * walsh_sign(q_head % 8, dim))
                                  : MLFloat16(
                                        0.00390625f *
                                        static_cast<float>((b * 5 + q_head * 3 + dim) % 11 - 5));
        }
      }
      for (int kv_head = 0; kv_head < kv_num_heads; ++kv_head) {
        for (int dim = 0; dim < head_size; ++dim) {
          const int index = (b * kv_num_heads + kv_head) * head_size + dim;
          key_data[index] = c.discriminating_attention
                                ? MLFloat16(0.48f * walsh_sign(past_seqlen % 8, dim))
                                : MLFloat16(
                                      key_step *
                                      static_cast<float>((b * 7 + kv_head * 5 + dim) % 13 - 6));
          value_data[index] = c.discriminating_attention
                                  ? MLFloat16(0.08f + 0.04f * static_cast<float>(past_seqlen % 8) +
                                              0.04f * static_cast<float>(dim % 2))
                                  : MLFloat16(
                                        value_step *
                                        static_cast<float>((b * 3 + kv_head * 7 + dim) % 17 - 8));
        }
      }
    }
    for (int block_id = 0; block_id < num_blocks; ++block_id) {
      for (int slot = 0; slot < block_size; ++slot) {
        for (int kv_head = 0; kv_head < kv_num_heads; ++kv_head) {
          for (int dim = 0; dim < head_size; ++dim) {
            const int index = CacheIndex(block_id, slot, kv_head, dim,
                                         block_size, kv_num_heads, head_size);
            if (c.discriminating_attention && slot == past_seqlen) {
              key_cache_data[index] = MLFloat16(0.0f);
              value_cache_data[index] = MLFloat16(0.04f);
            } else {
              key_cache_data[index] = c.discriminating_attention
                                          ? MLFloat16(0.48f * walsh_sign(slot % 8, dim))
                                          : MLFloat16(
                                                cache_key_step *
                                                static_cast<float>(
                                                    (block_id * 3 + slot + kv_head * 5 + dim) % 13 - 6));
              value_cache_data[index] =
                  c.discriminating_attention
                      ? MLFloat16(0.08f + 0.04f * static_cast<float>(slot % 8) +
                                  0.04f * static_cast<float>(dim % 2))
                      : MLFloat16(
                            cache_value_step *
                            static_cast<float>((block_id * 5 + slot * 3 + kv_head * 7 + dim) % 17 - 8));
            }
          }
        }
      }
    }
  } else if (c.split_sensitive_values) {
    const int sequence_capacity = max_num_blocks_per_seq * block_size;
    for (int b = 0; b < batch_size; ++b) {
      const float low_value = -0.1f + 0.2f * b;
      const float high_value = 0.1f + 0.2f * b;
      for (int slot = 0; slot < sequence_capacity; ++slot) {
        const MLFloat16 value(slot < past_seqlen / 2 ? low_value : high_value);
        const int block_id = block_table_data[b * max_num_blocks_per_seq + slot / block_size];
        const int slot_offset = CacheIndex(block_id, slot % block_size, 0, 0,
                                           block_size, kv_num_heads, head_size);
        std::fill_n(value_cache_data.begin() + slot_offset, kv_num_heads * head_size, value);
      }
    }
  }
  std::vector<BFloat16> query_data_bf16;
  std::vector<BFloat16> key_data_bf16;
  std::vector<BFloat16> value_data_bf16;
  if (c.bf16_query) {
    query_data_bf16.reserve(query_data.size());
    key_data_bf16.reserve(key_data.size());
    value_data_bf16.reserve(value_data.size());
    for (const auto value : query_data) query_data_bf16.emplace_back(value.ToFloat());
    for (const auto value : key_data) key_data_bf16.emplace_back(value.ToFloat());
    for (const auto value : value_data) value_data_bf16.emplace_back(value.ToFloat());
  }
  auto query_value = c.bf16_query
                         ? make_gpu(query_data_bf16, TensorShape({token_count, hidden_size}))
                         : make_gpu(query_data, TensorShape({token_count, hidden_size}));
  auto key_value = c.bf16_query
                       ? make_gpu(key_data_bf16, TensorShape({token_count, kv_hidden_size}))
                       : make_gpu(key_data, TensorShape({token_count, kv_hidden_size}));
  auto value_value = c.bf16_query
                         ? make_gpu(value_data_bf16, TensorShape({token_count, kv_hidden_size}))
                         : make_gpu(value_data, TensorShape({token_count, kv_hidden_size}));
  std::vector<int8_t> key_cache_int8;
  std::vector<int8_t> value_cache_int8;
#if !defined(DISABLE_FLOAT8_TYPES)
  std::vector<Float8E4M3FN> key_cache_fp8;
  std::vector<Float8E4M3FN> value_cache_fp8;
#endif
  std::vector<BFloat16> key_cache_bf16;
  std::vector<BFloat16> value_cache_bf16;
  OrtValue key_cache_value;
  OrtValue value_cache_value;
  if (c.int8_cache) {
    key_cache_int8.reserve(key_cache_data.size());
    value_cache_int8.reserve(value_cache_data.size());
    for (const auto value : key_cache_data) {
      key_cache_int8.push_back(static_cast<int8_t>(std::round(value.ToFloat() / cache_scale)));
    }
    for (const auto value : value_cache_data) {
      value_cache_int8.push_back(static_cast<int8_t>(std::round(value.ToFloat() / cache_scale)));
    }
    key_cache_value = make_gpu(key_cache_int8, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
    value_cache_value = make_gpu(value_cache_int8, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
#if !defined(DISABLE_FLOAT8_TYPES)
  } else if (c.fp8_cache) {
    key_cache_fp8.reserve(key_cache_data.size());
    value_cache_fp8.reserve(value_cache_data.size());
    for (const auto value : key_cache_data) key_cache_fp8.emplace_back(value.ToFloat() / cache_scale);
    for (const auto value : value_cache_data) value_cache_fp8.emplace_back(value.ToFloat() / cache_scale);
    key_cache_value = make_gpu(key_cache_fp8, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
    value_cache_value = make_gpu(value_cache_fp8, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
#endif
  } else if (c.bf16_query) {
    key_cache_bf16.reserve(key_cache_data.size());
    value_cache_bf16.reserve(value_cache_data.size());
    for (const auto value : key_cache_data) key_cache_bf16.emplace_back(value.ToFloat());
    for (const auto value : value_cache_data) value_cache_bf16.emplace_back(value.ToFloat());
    key_cache_value = make_gpu(key_cache_bf16, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
    value_cache_value = make_gpu(value_cache_bf16, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
  } else {
    key_cache_value = make_gpu(key_cache_data, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
    value_cache_value = make_gpu(value_cache_data, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
  }
  std::vector<int32_t> cumulative_sequence_length_data(batch_size + 1);
  std::iota(cumulative_sequence_length_data.begin(), cumulative_sequence_length_data.end(), 0);
  auto cumulative_sequence_length_value =
      make_gpu(cumulative_sequence_length_data, TensorShape({batch_size + 1}));
  std::vector<int32_t> past_seqlens_data =
      c.replay_past_seqlens.empty() ? std::vector<int32_t>(batch_size, past_seqlen)
                                    : c.replay_past_seqlens.front();
  auto past_seqlens_value = make_gpu(past_seqlens_data, TensorShape({batch_size}));
  auto block_table_value =
      make_gpu(block_table_data, TensorShape({batch_size, max_num_blocks_per_seq}));
  auto output_value = c.bf16_query
                          ? make_gpu(std::vector<BFloat16>(token_count * hidden_size),
                                     TensorShape({token_count, hidden_size}))
                          : make_gpu(std::vector<MLFloat16>(token_count * hidden_size),
                                     TensorShape({token_count, hidden_size}));
  OrtValue key_cache_out_value;
  OrtValue value_cache_out_value;
  if (c.int8_cache) {
    key_cache_out_value = make_gpu(key_cache_int8, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
    value_cache_out_value = make_gpu(value_cache_int8, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
#if !defined(DISABLE_FLOAT8_TYPES)
  } else if (c.fp8_cache) {
    key_cache_out_value = make_gpu(key_cache_fp8, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
    value_cache_out_value = make_gpu(value_cache_fp8, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
#endif
  } else if (c.bf16_query) {
    key_cache_out_value = make_gpu(key_cache_bf16, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
    value_cache_out_value = make_gpu(value_cache_bf16, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
  } else {
    key_cache_out_value = make_gpu(key_cache_data, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
    value_cache_out_value = make_gpu(value_cache_data, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
  }
  OrtValue k_scale_value;
  OrtValue v_scale_value;
  if (quantized_cache) {
    k_scale_value = make_gpu(std::vector<float>{cache_scale}, TensorShape({1}));
    v_scale_value = make_gpu(std::vector<float>{cache_scale}, TensorShape({1}));
  }

  std::unique_ptr<IOBinding> io_binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&io_binding));
  ASSERT_STATUS_OK(io_binding->BindInput("query", query_value));
  ASSERT_STATUS_OK(io_binding->BindInput("key", key_value));
  ASSERT_STATUS_OK(io_binding->BindInput("value", value_value));
  ASSERT_STATUS_OK(io_binding->BindInput("key_cache", key_cache_value));
  ASSERT_STATUS_OK(io_binding->BindInput("value_cache", value_cache_value));
  ASSERT_STATUS_OK(io_binding->BindInput("cumulative_sequence_length", cumulative_sequence_length_value));
  ASSERT_STATUS_OK(io_binding->BindInput("past_seqlens", past_seqlens_value));
  ASSERT_STATUS_OK(io_binding->BindInput("block_table", block_table_value));
  if (quantized_cache) {
    ASSERT_STATUS_OK(io_binding->BindInput("k_scale", k_scale_value));
    ASSERT_STATUS_OK(io_binding->BindInput("v_scale", v_scale_value));
  }
  if (!c.attention_metadata.empty()) {
    ASSERT_STATUS_OK(io_binding->BindInput("attention_metadata", attention_metadata_value));
  }
  ASSERT_STATUS_OK(io_binding->BindOutput("output", output_value));
  if (!omit_cache_outputs) {
    ASSERT_STATUS_OK(io_binding->BindOutput("key_cache_out", alias_cache_outputs ? key_cache_value : key_cache_out_value));
    ASSERT_STATUS_OK(io_binding->BindOutput("value_cache_out", alias_cache_outputs ? value_cache_value : value_cache_out_value));
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  const size_t run_count = c.replay_past_seqlens.empty() ? 1 : c.replay_past_seqlens.size();
  RunOptions run_options;
  if (c.enable_cuda_graph) {
    ASSERT_STATUS_OK(run_options.config_options.AddConfigEntry("gpu_graph_id", "1"));
  }
  for (size_t run_index = 0; run_index < run_count; ++run_index) {
    if (!c.replay_past_seqlens.empty()) {
      past_seqlens_data = c.replay_past_seqlens[run_index];
      Tensor cpu_past_seqlens(DataTypeImpl::GetType<int32_t>(), TensorShape({batch_size}),
                              past_seqlens_data.data(), cpu_alloc->Info());
      ORT_THROW_IF_ERROR(
          execution_provider_ptr->GetDataTransfer()->CopyTensor(cpu_past_seqlens,
                                                                *past_seqlens_value.GetMutable<Tensor>()));
      Tensor cpu_cumulative(DataTypeImpl::GetType<int32_t>(), TensorShape({batch_size + 1}),
                            cumulative_sequence_length_data.data(), cpu_alloc->Info());
      ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(
          cpu_cumulative, *cumulative_sequence_length_value.GetMutable<Tensor>()));
    }

    const Status run_status = session.Run(run_options, *io_binding);
    if (!c.expected_error.empty()) {
      EXPECT_FALSE(run_status.IsOK());
      EXPECT_NE(run_status.ErrorMessage().find(c.expected_error), std::string::npos)
          << run_status.ErrorMessage();
      return;
    }
    ASSERT_STATUS_OK(run_status);

    Tensor cpu_output(c.bf16_query ? DataTypeImpl::GetType<BFloat16>() : DataTypeImpl::GetType<MLFloat16>(),
                      TensorShape({token_count, hidden_size}), cpu_alloc);
    ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(output_value.Get<Tensor>(), cpu_output));
    for (int b = 0; b < batch_size; ++b) {
      const int new_slot = past_seqlens_data[b];
      const int new_block_id =
          block_table_data[b * max_num_blocks_per_seq + new_slot / block_size];
      for (int kv_head = 0; kv_head < kv_num_heads; ++kv_head) {
        for (int dim = 0; dim < head_size; ++dim) {
          const int cache_index = CacheIndex(new_block_id, new_slot % block_size, kv_head, dim,
                                             block_size, kv_num_heads, head_size);
          const int input_index = (b * kv_num_heads + kv_head) * head_size + dim;
          key_cache_data[cache_index] = key_data[input_index];
          value_cache_data[cache_index] = value_data[input_index];
        }
      }

      const int gqa_factor = num_heads / kv_num_heads;
      for (int q_head = 0; q_head < num_heads; ++q_head) {
        const int kv_head = q_head / gqa_factor;
        std::vector<float> scores(new_slot + 1);
        float max_score = -std::numeric_limits<float>::infinity();
        for (int slot = 0; slot <= new_slot; ++slot) {
          const int block_id =
              block_table_data[b * max_num_blocks_per_seq + slot / block_size];
          float dot = 0.0f;
          for (int dim = 0; dim < head_size; ++dim) {
            const int query_index = (b * num_heads + q_head) * head_size + dim;
            const int cache_index = CacheIndex(block_id, slot % block_size, kv_head, dim,
                                               block_size, kv_num_heads, head_size);
            dot += query_data[query_index].ToFloat() * key_cache_data[cache_index].ToFloat();
          }
          scores[slot] = dot * scale;
          max_score = std::max(max_score, scores[slot]);
        }
        float denominator = 0.0f;
        for (float& score : scores) {
          score = std::exp(score - max_score);
          denominator += score;
        }
        for (int dim = 0; dim < head_size; ++dim) {
          float numerator = 0.0f;
          for (int slot = 0; slot <= new_slot; ++slot) {
            const int block_id =
                block_table_data[b * max_num_blocks_per_seq + slot / block_size];
            const int cache_index = CacheIndex(block_id, slot % block_size, kv_head, dim,
                                               block_size, kv_num_heads, head_size);
            numerator += scores[slot] * value_cache_data[cache_index].ToFloat();
          }
          const int output_index = (b * num_heads + q_head) * head_size + dim;
          const float actual_output = c.bf16_query
                                          ? cpu_output.Data<BFloat16>()[output_index].ToFloat()
                                          : cpu_output.Data<MLFloat16>()[output_index].ToFloat();
          const float expected_output = numerator / denominator;
          if (c.discriminating_attention) {
            EXPECT_GT(std::abs(expected_output), 2e-2f)
                << "The H256 fixture must reject an all-zero attention output.";
          }
          EXPECT_NEAR(actual_output, expected_output, 2e-3f)
              << "run=" << run_index << ", batch=" << b
              << ", q_head=" << q_head << ", dim=" << dim;
        }
      }
    }
  }
  if (c.enable_cuda_graph) {
    EXPECT_TRUE(execution_provider_ptr->IsGraphCaptured(1));
  }

  const auto& outputs = io_binding->GetOutputs();
  MLDataType cache_data_type = c.int8_cache   ? DataTypeImpl::GetType<int8_t>()
                               : c.bf16_query ? DataTypeImpl::GetType<BFloat16>()
                                              : DataTypeImpl::GetType<MLFloat16>();
#if !defined(DISABLE_FLOAT8_TYPES)
  if (c.fp8_cache) {
    cache_data_type = DataTypeImpl::GetType<Float8E4M3FN>();
  }
#endif
  auto read_cache_value = [&](const Tensor& tensor, size_t offset) {
    if (c.int8_cache) {
      return tensor.Data<int8_t>()[offset] * cache_scale;
    }
#if !defined(DISABLE_FLOAT8_TYPES)
    if (c.fp8_cache) {
      return tensor.Data<Float8E4M3FN>()[offset].ToFloat() * cache_scale;
    }
#endif
    return c.bf16_query ? tensor.Data<BFloat16>()[offset].ToFloat()
                        : tensor.Data<MLFloat16>()[offset].ToFloat();
  };
  if (omit_cache_outputs) {
    Tensor cpu_key_cache(cache_data_type,
                         TensorShape({num_blocks, block_size, kv_num_heads, head_size}), cpu_alloc);
    Tensor cpu_value_cache(cache_data_type,
                           TensorShape({num_blocks, block_size, kv_num_heads, head_size}), cpu_alloc);
    ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(key_cache_value.Get<Tensor>(), cpu_key_cache));
    ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(value_cache_value.Get<Tensor>(), cpu_value_cache));
    for (int b = 0; b < batch_size; ++b) {
      const int block_id =
          block_table_data[b * max_num_blocks_per_seq + past_seqlen / block_size];
      const size_t cache_update_offset =
          static_cast<size_t>(CacheIndex(block_id, past_seqlen % block_size, 0, 0,
                                         block_size, kv_num_heads, head_size));
      EXPECT_NEAR(read_cache_value(cpu_key_cache, cache_update_offset), 0.03f, 1e-3f);
      EXPECT_NEAR(read_cache_value(cpu_value_cache, cache_update_offset), 0.04f + 0.02f * b, 1e-3f);
    }
    ASSERT_EQ(outputs.size(), 1u);
    return;
  }

  ASSERT_EQ(outputs.size(), 3u);
  if (alias_cache_outputs) {
    EXPECT_EQ(outputs[1].Get<Tensor>().DataRaw(), key_cache_value.Get<Tensor>().DataRaw());
    EXPECT_EQ(outputs[2].Get<Tensor>().DataRaw(), value_cache_value.Get<Tensor>().DataRaw());
  } else {
    EXPECT_NE(outputs[1].Get<Tensor>().DataRaw(), key_cache_value.Get<Tensor>().DataRaw());
    EXPECT_NE(outputs[2].Get<Tensor>().DataRaw(), value_cache_value.Get<Tensor>().DataRaw());
  }

  // Verify K/V scatter actually landed at slot `past_seqlen` in both caches.
  // The input caches are initialized to 0.01 / 0.02; the new K/V are 0.03 / 0.04.
  // Without this, a scatter regression on either path (alias or non-alias)
  // would still leave `output[0]` non-zero and pointer identity intact, so
  // the smoke-only version of this test could not distinguish "scatter ran"
  // from "scatter silently didn't run". Downloading from the bound output
  // tensors covers both the aliased path (output backed by the input cache
  // buffer) and the non-aliased path (output backed by a separate buffer).
  Tensor cpu_key_cache_out(cache_data_type,
                           TensorShape({num_blocks, block_size, kv_num_heads, head_size}), cpu_alloc);
  Tensor cpu_value_cache_out(cache_data_type,
                             TensorShape({num_blocks, block_size, kv_num_heads, head_size}), cpu_alloc);
  ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(outputs[1].Get<Tensor>(), cpu_key_cache_out));
  ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(outputs[2].Get<Tensor>(), cpu_value_cache_out));
  for (int b = 0; b < batch_size; ++b) {
    const int last_past_seqlen = past_seqlens_data[b];
    const int block_id =
        block_table_data[b * max_num_blocks_per_seq + last_past_seqlen / block_size];
    for (int kv_head = 0; kv_head < kv_num_heads; ++kv_head) {
      for (int dim = 0; dim < head_size; ++dim) {
        const size_t cache_update_offset =
            static_cast<size_t>(CacheIndex(block_id, last_past_seqlen % block_size, kv_head, dim,
                                           block_size, kv_num_heads, head_size));
        const size_t input_offset = static_cast<size_t>((b * kv_num_heads + kv_head) * head_size + dim);
        EXPECT_NEAR(read_cache_value(cpu_key_cache_out, cache_update_offset),
                    key_data[input_offset].ToFloat(), 1e-3f)
            << "batch=" << b << ", kv_head=" << kv_head << ", dim=" << dim;
        EXPECT_NEAR(read_cache_value(cpu_value_cache_out, cache_update_offset),
                    value_data[input_offset].ToFloat(), 1e-3f)
            << "batch=" << b << ", kv_head=" << kv_head << ", dim=" << dim;
      }
    }
  }
}

void RunEndToEndCaseOnAvailableProviders(const EndToEndCase& c) {
  bool ran = false;

  // CUDA PagedAttention requires key_cache/value_cache outputs to alias the
  // corresponding input buffers. OpTester allocates distinct output buffers,
  // so this harness cannot satisfy the CUDA aliasing contract.
  // Keep the tests shared here and run them on WebGPU, where non-aliased
  // output buffers are explicitly supported by the kernel fallback path.

  if (auto webgpu_ep = DefaultWebGpuExecutionProvider()) {
    RunEndToEndCase(c, std::move(webgpu_ep));
    ran = true;
  }

  if (!ran) {
    GTEST_SKIP() << "No PagedAttention execution provider is available.";
  }
}

}  // namespace

TEST(PagedAttention, Cuda_AliasedCache_IOBinding) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true);
}

TEST(PagedAttention, Cuda_AttentionMetadataShape2CompatibilityAndDispatchBounds) {
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "0"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  IoBindingCase c;
  c.attention_metadata = {1, 256};

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();

  EXPECT_NE(debug_output.find("Operator=PagedAttention"), std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("SdpaKernel=DECODER_ATTENTION"), std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("NumSplits=2"), std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("GqaGroupSize=1"), std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("EffectiveKvLengthBound=256"), std::string::npos) << debug_output;
}

struct XqaHeadSize256TypeCase {
  bool bf16_query;
  bool fp8_cache;
  const char* name;
};

std::vector<XqaHeadSize256TypeCase> XqaHeadSize256TypeCases() {
  std::vector<XqaHeadSize256TypeCase> cases{
      {false, false, "FP16_INT8"},
      {true, false, "BF16_INT8"},
  };
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
  cases.push_back({false, true, "FP16_FP8"});
  cases.push_back({true, true, "BF16_FP8"});
#endif
  return cases;
}

class PagedAttentionXqaHeadSize256Test
    : public ::testing::TestWithParam<XqaHeadSize256TypeCase> {};

TEST_P(PagedAttentionXqaHeadSize256Test, Group6DispatchAndParity) {
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "1"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "0"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"},
          {"ORT_ENABLE_XQA", "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  const auto& type_case = GetParam();
  const int cuda_architecture = GetCudaArchitecture();
  if (cuda_architecture < 800) {
    GTEST_SKIP() << "XQA requires compute capability 8.0 or later.";
  }
  if (type_case.fp8_cache && cuda_architecture < 890) {
    GTEST_SKIP() << "FP8 XQA requires compute capability 8.9 or later.";
  }

  auto make_case = [&](int group_size) {
    IoBindingCase c;
    c.num_heads = group_size;
    c.kv_num_heads = 1;
    c.head_size = 256;
    c.num_blocks = 2;
    c.max_num_blocks_per_seq = 1;
    c.past_seqlen = 122;
    c.int8_cache = !type_case.fp8_cache;
    c.fp8_cache = type_case.fp8_cache;
    c.bf16_query = type_case.bf16_query;
    c.irregular_layout = true;
    c.discriminating_attention = true;
    c.attention_metadata = {1, 128, 128};
    return c;
  };

  // Group 4 uses the same H256 dtype family and shared-memory geometry as group 6. If it cannot
  // dispatch XQA, this build/device lacks a runnable specialization and the fallback is expected.
  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, make_case(4));
  const std::string baseline_debug_output = testing::internal::GetCapturedStdout();
  if (baseline_debug_output.find("SdpaKernel=XQA") == std::string::npos) {
    GTEST_SKIP() << "Paged XQA H256 " << type_case.name
                 << " is not runnable in this build/device configuration.\n"
                 << baseline_debug_output;
  }

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, make_case(6));
  const std::string debug_output = testing::internal::GetCapturedStdout();
  EXPECT_NE(debug_output.find("SdpaKernel=XQA"), std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("GqaGroupSize=6"), std::string::npos) << debug_output;
}

INSTANTIATE_TEST_SUITE_P(
    QueryAndCacheTypes,
    PagedAttentionXqaHeadSize256Test,
    ::testing::ValuesIn(XqaHeadSize256TypeCases()),
    [](const ::testing::TestParamInfo<XqaHeadSize256TypeCase>& info) {
      return info.param.name;
    });

bool IsNativeFp16HeadSize256Group6XqaRunnable() {
  IoBindingCase c;
  c.num_heads = 6;
  c.kv_num_heads = 1;
  c.head_size = 256;
  c.num_blocks = 2;
  c.max_num_blocks_per_seq = 1;
  c.past_seqlen = 122;
  c.attention_metadata = {1, 128, 128};

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();
  return debug_output.find("SdpaKernel=XQA") != std::string::npos;
}

TEST(PagedAttention, Cuda_XqaNativeFp16CacheHeadSize256Group6) {
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "0"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"},
          {"ORT_ENABLE_XQA", "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "XQA requires compute capability 8.0 or later.";
  }
  if (!IsNativeFp16HeadSize256Group6XqaRunnable()) {
    GTEST_SKIP() << "Native FP16 paged XQA H256/group6 is not runnable in this build/device configuration.";
  }

  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  provider_options.enable_cuda_graph = true;

  IoBindingCase c;
  c.batch_size = 2;
  c.num_heads = 6;
  c.kv_num_heads = 1;
  c.head_size = 256;
  c.num_blocks = 32;
  c.max_num_blocks_per_seq = 8;
  c.past_seqlen = 2047;
  c.enable_cuda_graph = true;
  c.irregular_layout = true;
  c.discriminating_attention = true;
  c.replay_past_seqlens = {{1024, 1536}, {1536, 2047}, {2047, 1024}};
  c.block_table = {7, 2, 11, 4, 15, 0, 9, 6,
                   23, 18, 27, 20, 31, 16, 25, 22};
  c.attention_metadata = {1, 2048, 1025};

  testing::internal::CaptureStdout();
  RunIoBindingCase(CudaExecutionProviderWithOptions(&provider_options),
                   kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();

  const std::string xqa_backend = "SdpaKernel=XQA";
  const size_t first_xqa = debug_output.find(xqa_backend);
  const size_t second_xqa =
      first_xqa == std::string::npos ? std::string::npos : debug_output.find(xqa_backend, first_xqa + 1);
  EXPECT_NE(first_xqa, std::string::npos) << debug_output;
  EXPECT_NE(second_xqa, std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("GqaGroupSize=6"), std::string::npos) << debug_output;
}

TEST(PagedAttention, Cuda_XqaNativeFp16CacheSplitReduction) {
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "0"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"},
          {"ORT_ENABLE_XQA", "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "XQA requires compute capability 8.0 or later.";
  }
  if (!IsNativeFp16HeadSize256Group6XqaRunnable()) {
    GTEST_SKIP() << "Native FP16 paged XQA H256/group6 is not runnable in this build/device configuration.";
  }

  IoBindingCase c;
  c.num_heads = 6;
  c.kv_num_heads = 1;
  c.head_size = 256;
  c.num_blocks = 16;
  c.max_num_blocks_per_seq = 8;
  c.past_seqlen = 2047;
  c.split_sensitive_values = true;
  c.attention_metadata = {1, 2048, 2048};

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();
  EXPECT_NE(debug_output.find("SdpaKernel=XQA"), std::string::npos) << debug_output;
}

TEST(PagedAttention, Cuda_XqaNativeFp16CacheFallsBackWithoutMetadata) {
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "0"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  IoBindingCase c;
  c.num_heads = 6;
  c.kv_num_heads = 1;
  c.head_size = 256;
  c.num_blocks = 16;
  c.max_num_blocks_per_seq = 8;
  c.past_seqlen = 2047;

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(debug_output.find("SdpaKernel=XQA"), std::string::npos) << debug_output;
  EXPECT_TRUE(debug_output.find("SdpaKernel=FLASH_ATTENTION") != std::string::npos ||
              debug_output.find("SdpaKernel=DECODER_ATTENTION") != std::string::npos)
      << debug_output;
}

TEST(PagedAttention, Cuda_XqaNativeFp16CacheMultiTokenBoundFallsBack) {
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "0"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "0"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  IoBindingCase c;
  c.batch_size = 2;
  c.num_heads = 6;
  c.kv_num_heads = 1;
  c.head_size = 256;
  c.num_blocks = 32;
  c.max_num_blocks_per_seq = 8;
  c.past_seqlen = 2047;
  c.attention_metadata = {2, 2048, 2048};

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(debug_output.find("SdpaKernel=XQA"), std::string::npos) << debug_output;
  EXPECT_TRUE(debug_output.find("SdpaKernel=FLASH_ATTENTION") != std::string::npos ||
              debug_output.find("SdpaKernel=DECODER_ATTENTION") != std::string::npos)
      << debug_output;
}

TEST(PagedAttention, Cuda_AttentionMetadataValidation) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  struct InvalidMetadataCase {
    std::vector<int32_t> metadata;
    const char* expected_error;
  };
  const std::vector<InvalidMetadataCase> cases = {
      {{1, 256, -1}, "entries must be non-negative"},
      {{1, 128, 129}, "must not exceed max_kv_len_bound"},
      {{1, 0, 257}, "must not exceed max_kv_len_bound"},
      {{1, 256, 128, 64}, "must have shape (2) or (3)"},
  };

  for (const auto& test_case : cases) {
    IoBindingCase c;
    c.attention_metadata = test_case.metadata;
    c.expected_error = test_case.expected_error;
    RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  }
}

TEST(PagedAttention, Cuda_FlashSplitKvLongContext) {
#if defined(USE_FLASH_ATTENTION)
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "1"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Flash Attention requires compute capability 8.0 or later.";
  }

  IoBindingCase c;
  c.batch_size = 2;
  c.num_heads = 4;
  c.kv_num_heads = 2;
  c.head_size = 128;
  c.num_blocks = 32;
  c.max_num_blocks_per_seq = 16;
  c.irregular_layout = true;
  c.replay_past_seqlens = {{767, 2047}};
  c.block_table.resize(c.batch_size * c.max_num_blocks_per_seq);
  for (int i = 0; i < static_cast<int>(c.block_table.size()); ++i) {
    c.block_table[i] = (i * 13 + 7) % c.num_blocks;
  }
  c.attention_metadata = {1, 4096, 2048};

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();

  EXPECT_NE(debug_output.find("SdpaKernel=FLASH_ATTENTION"), std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("EffectiveKvLengthBound=4096"), std::string::npos) << debug_output;
  const std::string split_prefix = "NumSplits=";
  const size_t split_pos = debug_output.find(split_prefix);
  ASSERT_NE(split_pos, std::string::npos) << debug_output;
  EXPECT_GT(std::stoi(debug_output.substr(split_pos + split_prefix.size())), 1) << debug_output;
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(PagedAttention, Cuda_FlashSplitKvCudaGraphReplay) {
#if defined(USE_FLASH_ATTENTION)
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "1"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Flash Attention requires compute capability 8.0 or later.";
  }

  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  provider_options.enable_cuda_graph = true;

  IoBindingCase c;
  c.batch_size = 2;
  c.num_heads = 2;
  c.kv_num_heads = 1;
  c.head_size = 128;
  c.num_blocks = 256;
  c.max_num_blocks_per_seq = 128;
  c.irregular_layout = true;
  c.enable_cuda_graph = true;
  c.replay_past_seqlens = {
      {512, 512},
      {513, 1024},
      {1024, 4096},
      {2048, 8192},
  };
  c.attention_metadata = {1, 32768, 513};

  testing::internal::CaptureStdout();
  RunIoBindingCase(CudaExecutionProviderWithOptions(&provider_options),
                   kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();

  EXPECT_NE(debug_output.find("SdpaKernel=FLASH_ATTENTION"), std::string::npos) << debug_output;
  const std::string split_prefix = "NumSplits=";
  const size_t split_pos = debug_output.find(split_prefix);
  ASSERT_NE(split_pos, std::string::npos) << debug_output;
  EXPECT_GT(std::stoi(debug_output.substr(split_pos + split_prefix.size())), 1) << debug_output;
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(PagedAttention, CudaGraphRequiresAttentionMetadata) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.enable_cuda_graph = true;

  IoBindingCase c;
  c.expected_initialization_error = "requires PagedAttention input 'attention_metadata'";
  RunIoBindingCase(CudaExecutionProviderWithOptions(&provider_options),
                   kCudaExecutionProvider, true, false, c);
}

TEST(PagedAttention, Cuda_FlashSplitKvInt8Cache) {
#if defined(USE_FLASH_ATTENTION)
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "1"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Flash Attention requires compute capability 8.0 or later.";
  }

  IoBindingCase c;
  c.batch_size = 2;
  c.num_heads = 2;
  c.kv_num_heads = 1;
  c.head_size = 128;
  c.num_blocks = 32;
  c.max_num_blocks_per_seq = 16;
  c.past_seqlen = 2047;
  c.split_sensitive_values = true;
  c.int8_cache = true;
  c.attention_metadata = {1, 2048, 2048};

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();

  EXPECT_NE(debug_output.find("SdpaKernel=FLASH_ATTENTION"), std::string::npos) << debug_output;
  const std::string split_prefix = "NumSplits=";
  const size_t split_pos = debug_output.find(split_prefix);
  ASSERT_NE(split_pos, std::string::npos) << debug_output;
  EXPECT_GT(std::stoi(debug_output.substr(split_pos + split_prefix.size())), 1) << debug_output;
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(PagedAttention, Cuda_FlashSplitKvSkipsShortReplayRange) {
#if defined(USE_FLASH_ATTENTION)
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::attention::kDisableFlashAttention, "0"},
          {onnxruntime::contrib::attention::kDisableMemoryEfficientAttention, "1"},
          {onnxruntime::contrib::attention::kDisableDecoderAttention, "1"},
          {onnxruntime::contrib::attention::kEnableAttentionKernelDebugInfo, "1"}}};

  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Flash Attention requires compute capability 8.0 or later.";
  }

  IoBindingCase c;
  c.num_heads = 2;
  c.kv_num_heads = 1;
  c.head_size = 128;
  c.num_blocks = 16;
  c.max_num_blocks_per_seq = 16;
  c.past_seqlen = 127;
  c.attention_metadata = {1, 2048, 128};

  testing::internal::CaptureStdout();
  RunIoBindingCase(DefaultCudaExecutionProvider(), kCudaExecutionProvider, true, false, c);
  const std::string debug_output = testing::internal::GetCapturedStdout();

  EXPECT_NE(debug_output.find("SdpaKernel=FLASH_ATTENTION"), std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("EffectiveKvLengthBound=2048"), std::string::npos) << debug_output;
  EXPECT_NE(debug_output.find("NumSplits=1"), std::string::npos) << debug_output;
#else
  GTEST_SKIP() << "Flash Attention is not enabled in this build.";
#endif
}

TEST(PagedAttention, WebGpu_AliasedCache_IOBinding) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }
  RunIoBindingCase(DefaultWebGpuExecutionProvider(), kWebGpuExecutionProvider, true);
}

TEST(PagedAttention, WebGpu_NonAliasedCache_IOBinding) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }
  RunIoBindingCase(DefaultWebGpuExecutionProvider(), kWebGpuExecutionProvider, false);
}

TEST(PagedAttention, WebGpu_OmittedCacheOutputs_IOBinding) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }
  RunIoBindingCase(DefaultWebGpuExecutionProvider(), kWebGpuExecutionProvider, false, true);
}

TEST(PagedAttention, WebGpu_RejectsShortRotaryCache) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  constexpr int token_count = 1;
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 16;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int block_size = 256;
  constexpr int past_seqlen = 4;
  constexpr int rotary_cache_length = 4;
  constexpr int cache_elems = block_size * kv_num_heads * head_size;

  OpTester test("PagedAttention", 1, kMSDomain);
  test.AddAttribute<int64_t>("num_heads", num_heads);
  test.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
  test.AddAttribute<int64_t>("do_rotary", 1);
  test.AddInput<MLFloat16>("query", {token_count, hidden_size}, std::vector<MLFloat16>(hidden_size, MLFloat16(0.02f)));
  test.AddInput<MLFloat16>("key", {token_count, hidden_size}, std::vector<MLFloat16>(hidden_size, MLFloat16(0.03f)));
  test.AddInput<MLFloat16>("value", {token_count, hidden_size}, std::vector<MLFloat16>(hidden_size, MLFloat16(0.04f)));
  test.AddInput<MLFloat16>("key_cache", {1, block_size, kv_num_heads, head_size},
                           std::vector<MLFloat16>(cache_elems, MLFloat16(0.01f)));
  test.AddInput<MLFloat16>("value_cache", {1, block_size, kv_num_heads, head_size},
                           std::vector<MLFloat16>(cache_elems, MLFloat16(0.02f)));
  test.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, token_count});
  test.AddInput<int32_t>("past_seqlens", {1}, {past_seqlen});
  test.AddInput<int32_t>("block_table", {1, 1}, {0});
  test.AddInput<MLFloat16>("cos_cache", {rotary_cache_length, head_size / 2},
                           std::vector<MLFloat16>(rotary_cache_length * head_size / 2, MLFloat16(1.0f)));
  test.AddInput<MLFloat16>("sin_cache", {rotary_cache_length, head_size / 2},
                           std::vector<MLFloat16>(rotary_cache_length * head_size / 2, MLFloat16(0.0f)));
  test.AddOutput<MLFloat16>("output", {token_count, hidden_size}, std::vector<MLFloat16>(hidden_size));
  test.AddOutput<MLFloat16>("key_cache_out", {1, block_size, kv_num_heads, head_size},
                            std::vector<MLFloat16>(cache_elems));
  test.AddOutput<MLFloat16>("value_cache_out", {1, block_size, kv_num_heads, head_size},
                            std::vector<MLFloat16>(cache_elems));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectFailure, "cos_cache dimension 0", {}, nullptr, &execution_providers);
}

// Decode tier: single batch, single new Q token, non-zero past. The FA
// tier selector uses `sequence_length_ < 32` → split-reduce path.
TEST(PagedAttention, EndToEnd_Decode_SingleBatch_WithPast) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 1;
  c.cumulative_seqlens_q = {0, 1};
  c.past_seqlens = {4};  // total_kv_len = 5
  c.block_table = {0};
  RunEndToEndCaseOnAvailableProviders(c);
}

// Prefill tier: single batch, multiple new Q tokens, zero past. Exercises
// causal masking across new tokens.
TEST(PagedAttention, EndToEnd_Prefill_SingleBatch_NoPast) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 4;
  c.cumulative_seqlens_q = {0, 4};
  c.past_seqlens = {0};
  c.block_table = {0};
  RunEndToEndCaseOnAvailableProviders(c);
}

TEST(PagedAttention, EndToEnd_Prefill_PagedFlashAttention) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 32;
  c.cumulative_seqlens_q = {0, 32};
  c.past_seqlens = {0};
  c.block_table = {0};
  RunEndToEndCaseOnAvailableProviders(c);
}

// Multi-batch decode with differing past lengths — exercises variable-length
// packing across batches.
TEST(PagedAttention, EndToEnd_Decode_MultiBatch_VariablePast) {
  EndToEndCase c{};
  c.batch_size = 2;
  c.token_count = 2;  // one new Q per batch
  c.num_blocks = 3;
  c.cumulative_seqlens_q = {0, 1, 2};
  c.past_seqlens = {3, 6};  // total_kv_len = 4 and 7 respectively
  c.block_table = {0, 2};   // seq 0 → block 0, seq 1 → block 2
  RunEndToEndCaseOnAvailableProviders(c);
}

// GQA (num_heads > kv_num_heads): broadcasts each KV head across gqa_factor
// query heads. Verifies the head-index mapping matches FA's convention.
TEST(PagedAttention, EndToEnd_Decode_GQA) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 1;
  c.num_heads = 2;
  c.kv_num_heads = 1;  // GQA broadcast factor = 2
  c.cumulative_seqlens_q = {0, 1};
  c.past_seqlens = {3};
  c.block_table = {0};
  RunEndToEndCaseOnAvailableProviders(c);
}

// Empty-token fast path: the output is empty and cache outputs must still
// preserve the input cache contents in OpTester's non-aliased allocation path.
TEST(PagedAttention, EndToEnd_EmptyTokens_CopyCaches) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 0;
  c.cumulative_seqlens_q = {0, 0};
  c.past_seqlens = {0};
  c.block_table = {0};
  RunEndToEndCaseOnAvailableProviders(c);
}

// Mixed variable-length prefill (seq 0: 3 tokens, seq 1: 2 tokens) with
// non-zero past on both — the most realistic paged-attention scenario.
TEST(PagedAttention, EndToEnd_MixedPrefillDecode_MultiBatch_VariablePast) {
  EndToEndCase c{};
  c.batch_size = 2;
  c.token_count = 5;
  c.num_heads = 2;
  c.kv_num_heads = 1;  // GQA
  c.num_blocks = 3;
  c.cumulative_seqlens_q = {0, 3, 5};
  c.past_seqlens = {2, 4};
  c.block_table = {0, 2};
  RunEndToEndCaseOnAvailableProviders(c);
}

// Varlen prefill above the fused-paged-prefill gate: max_seqlen_q >= 32 and
// B * max_seqlen_q != token_count. Exercises the WebGPU fused shader's
// q_varlen=1 code path (raw-packed Q read via cumulative_seqlens_q). Uses
// head_size = 128 to match the shared-memory tile the shader is sized for.
TEST(PagedAttention, EndToEnd_Prefill_MultiBatch_Varlen_Fused) {
  EndToEndCase c{};
  c.batch_size = 2;
  c.token_count = 48;  // seq 0: 32 tokens, seq 1: 16 tokens
  c.num_heads = 2;
  c.kv_num_heads = 2;  // MHA (n_reps = 1)
  c.head_size = 128;
  c.num_blocks = 3;
  c.cumulative_seqlens_q = {0, 32, 48};
  c.past_seqlens = {0, 0};
  c.block_table = {0, 2};
  RunEndToEndCaseOnAvailableProviders(c);
}

// Fused paged prefill across a block boundary. block_size == max_k_step (32
// for fp16 head_size <= 128) is the tightest configuration
// ShouldRunFusedPagedPrefill accepts. token_count = 64 spans two paged
// blocks, and block_table = {3, 1} makes the two logical pages map to
// non-adjacent physical blocks — any indexing bug that assumes physical
// adjacency of consecutive logical pages will fail here (see the alignment
// invariant note at the top of flash_attention_paged_prefill.wgsl.template).
TEST(PagedAttention, EndToEnd_Prefill_FusedPrefill_BlockBoundaryCrossing) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 64;  // >= 32 for fused prefill; spans 2 blocks of 32
  c.num_heads = 2;
  c.kv_num_heads = 2;  // MHA
  c.head_size = 128;
  c.block_size = 32;
  c.num_blocks = 5;
  c.max_num_blocks_per_seq = 2;
  c.cumulative_seqlens_q = {0, 64};
  c.past_seqlens = {0};
  c.block_table = {3, 1};  // logical page 0 -> phys 3, page 1 -> phys 1
  RunEndToEndCaseOnAvailableProviders(c);
}

// Direct paged decode with a KV history that crosses a block boundary. Uses
// block_size = 32 and past = 40 so total KV = 41 spans two paged blocks;
// block_table = {4, 1} makes the two logical pages non-contiguous, catching
// any indexing bug in the per-slot block_table lookup inside
// flash_attention_paged_decode_qkv.wgsl.template.
TEST(PagedAttention, EndToEnd_Decode_BlockBoundaryCrossing) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 1;  // decode: one new Q token
  c.num_heads = 1;
  c.kv_num_heads = 1;
  c.head_size = 8;
  c.block_size = 32;
  c.num_blocks = 5;
  c.max_num_blocks_per_seq = 2;
  c.cumulative_seqlens_q = {0, 1};
  c.past_seqlens = {40};   // total KV = 41 slots, spans 2 blocks of 32
  c.block_table = {4, 1};  // logical page 0 -> phys 4, page 1 -> phys 1
  RunEndToEndCaseOnAvailableProviders(c);
}

// Force the ShouldRunFusedPagedPrefill *reject* branch. With fp16 head_size
// 128, max_k_step is 32, so block_size 16 fails the block_size >= max_k_step
// guard. PagedAttention must then take the gather-then-flash cascade (the
// merged #31611 code path). Verifies output parity against the reference in
// the fallback path, locking down the reject-boundary so a future refactor
// of ShouldRunFusedPagedPrefill can't silently change it.
TEST(PagedAttention, EndToEnd_Prefill_ForcedFallback_BlockSizeBelowMaxKStep) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 64;  // prefill (>= 32)
  c.num_heads = 2;
  c.kv_num_heads = 2;  // MHA
  c.head_size = 128;
  c.block_size = 16;  // < max_k_step (=32 for fp16 head_size<=128) => rejected
  c.num_blocks = 8;
  c.max_num_blocks_per_seq = 4;  // 64 tokens / 16 slots per block
  c.cumulative_seqlens_q = {0, 64};
  c.past_seqlens = {0};
  c.block_table = {3, 1, 5, 2};  // non-contiguous physical pages
  RunEndToEndCaseOnAvailableProviders(c);
}

// Fused paged prefill under GQA + nonzero past, with non-contiguous physical
// pages. Combines conditions the existing fused-prefill tests cover
// separately: fused path (max_seqlen_q >= 32), num_heads > kv_num_heads,
// past_seqlens != 0, and a block_table whose physical page order does not
// match the logical order. Locks the fused shader's kv_head_idx =
// head_idx / uniforms.n_reps mapping and the causal-mask past-offset
// derivation on the same shape.
TEST(PagedAttention, EndToEnd_Prefill_FusedPrefill_GQA_WithPast) {
  EndToEndCase c{};
  c.batch_size = 1;
  c.token_count = 32;  // prefill: max_seqlen_q >= 32 fires fused shader
  c.num_heads = 4;
  c.kv_num_heads = 2;  // GQA (n_reps = 2)
  c.head_size = 128;
  c.block_size = 32;  // == max_k_step for fp16 head_size <= 128 (min alignment)
  c.num_blocks = 6;
  c.max_num_blocks_per_seq = 3;  // covers total KV = 64 tokens across 2 blocks
  c.cumulative_seqlens_q = {0, 32};
  c.past_seqlens = {32};      // nonzero past
  c.block_table = {5, 2, 0};  // non-contiguous physical pages
  RunEndToEndCaseOnAvailableProviders(c);
}

}  // namespace test
}  // namespace onnxruntime
