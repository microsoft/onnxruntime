// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Operator-level correctness tests for PagedAttention. Each test compares the
// provider output and updated caches against a CPU scaled-dot-product-attention
// reference, and runs against each provider available in the build.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "default_providers.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
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
                      bool omit_cache_outputs = false) {
  constexpr int batch_size = 1;
  constexpr int token_count = 1;
  constexpr int num_heads = 1;
  constexpr int kv_num_heads = 1;
  constexpr int head_size = 8;
  constexpr int block_size = 256;
  constexpr int num_blocks = 2;
  constexpr int past_seqlen = 4;
  constexpr int hidden_size = num_heads * head_size;
  constexpr int cache_elems = num_blocks * block_size * kv_num_heads * head_size;

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

  auto& query_arg = graph.GetOrCreateNodeArg("query", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                                                      {token_count, hidden_size}));
  auto& key_arg = graph.GetOrCreateNodeArg("key", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                                                  {token_count, hidden_size}));
  auto& value_arg = graph.GetOrCreateNodeArg("value", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                                                      {token_count, hidden_size}));
  auto& key_cache_arg = graph.GetOrCreateNodeArg(
      "key_cache", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                   {num_blocks, block_size, kv_num_heads, head_size}));
  auto& value_cache_arg = graph.GetOrCreateNodeArg(
      "value_cache", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                     {num_blocks, block_size, kv_num_heads, head_size}));
  auto& cumulative_sequence_length_arg = graph.GetOrCreateNodeArg(
      "cumulative_sequence_length", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch_size + 1}));
  auto& past_seqlens_arg = graph.GetOrCreateNodeArg(
      "past_seqlens", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch_size}));
  auto& block_table_arg = graph.GetOrCreateNodeArg(
      "block_table", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch_size, 1}));
  auto& empty_optional_arg = graph.GetOrCreateNodeArg("", nullptr);
  std::vector<NodeArg*> input_defs = {&query_arg, &key_arg, &value_arg, &key_cache_arg, &value_cache_arg,
                                      &cumulative_sequence_length_arg, &past_seqlens_arg, &block_table_arg,
                                      &empty_optional_arg, &empty_optional_arg};

  auto& output_arg = graph.GetOrCreateNodeArg(
      "output", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {token_count, hidden_size}));
  auto& key_cache_out_arg = graph.GetOrCreateNodeArg(
      "key_cache_out", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                       {num_blocks, block_size, kv_num_heads, head_size}));
  auto& value_cache_out_arg = graph.GetOrCreateNodeArg(
      "value_cache_out", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
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
  ASSERT_STATUS_OK(session.Initialize());
  auto device_alloc = session.GetAllocator(device_memory_info);
  ASSERT_NE(device_alloc, nullptr);

  auto cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto make_gpu_fp16 = [&](const std::vector<MLFloat16>& data, const TensorShape& shape) {
    Tensor cpu_tensor(DataTypeImpl::GetType<MLFloat16>(), shape, const_cast<MLFloat16*>(data.data()), cpu_alloc->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<MLFloat16>(), shape, device_alloc);
    ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue value;
    Tensor::InitOrtValue(std::move(gpu_tensor), value);
    return value;
  };
  auto make_gpu_int32 = [&](const std::vector<int32_t>& data, const TensorShape& shape) {
    Tensor cpu_tensor(DataTypeImpl::GetType<int32_t>(), shape, const_cast<int32_t*>(data.data()), cpu_alloc->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<int32_t>(), shape, device_alloc);
    ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue value;
    Tensor::InitOrtValue(std::move(gpu_tensor), value);
    return value;
  };

  std::vector<MLFloat16> query_data(token_count * hidden_size, MLFloat16(0.02f));
  std::vector<MLFloat16> key_data(token_count * hidden_size, MLFloat16(0.03f));
  std::vector<MLFloat16> value_data(token_count * hidden_size, MLFloat16(0.04f));
  std::vector<MLFloat16> key_cache_data(cache_elems, MLFloat16(0.01f));
  std::vector<MLFloat16> value_cache_data(cache_elems, MLFloat16(0.02f));
  auto query_value = make_gpu_fp16(query_data, TensorShape({token_count, hidden_size}));
  auto key_value = make_gpu_fp16(key_data, TensorShape({token_count, hidden_size}));
  auto value_value = make_gpu_fp16(value_data, TensorShape({token_count, hidden_size}));
  auto key_cache_value = make_gpu_fp16(key_cache_data, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
  auto value_cache_value = make_gpu_fp16(value_cache_data, TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
  auto cumulative_sequence_length_value = make_gpu_int32({0, token_count}, TensorShape({batch_size + 1}));
  auto past_seqlens_value = make_gpu_int32({past_seqlen}, TensorShape({batch_size}));
  auto block_table_value = make_gpu_int32({0}, TensorShape({batch_size, 1}));
  auto output_value = make_gpu_fp16(std::vector<MLFloat16>(token_count * hidden_size),
                                    TensorShape({token_count, hidden_size}));
  auto key_cache_out_value = make_gpu_fp16(key_cache_data,
                                           TensorShape({num_blocks, block_size, kv_num_heads, head_size}));
  auto value_cache_out_value = make_gpu_fp16(value_cache_data,
                                             TensorShape({num_blocks, block_size, kv_num_heads, head_size}));

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
  ASSERT_STATUS_OK(io_binding->BindOutput("output", output_value));
  if (!omit_cache_outputs) {
    ASSERT_STATUS_OK(io_binding->BindOutput("key_cache_out", alias_cache_outputs ? key_cache_value : key_cache_out_value));
    ASSERT_STATUS_OK(io_binding->BindOutput("value_cache_out", alias_cache_outputs ? value_cache_value : value_cache_out_value));
  }

  RunOptions run_options;
  ASSERT_STATUS_OK(session.Run(run_options, *io_binding));

  Tensor cpu_output(DataTypeImpl::GetType<MLFloat16>(), TensorShape({token_count, hidden_size}), cpu_alloc);
  ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(output_value.Get<Tensor>(), cpu_output));
  EXPECT_NE(cpu_output.Data<MLFloat16>()[0].ToFloat(), 0.0f);

  const auto& outputs = io_binding->GetOutputs();
  if (omit_cache_outputs) {
    Tensor cpu_key_cache(DataTypeImpl::GetType<MLFloat16>(),
                         TensorShape({num_blocks, block_size, kv_num_heads, head_size}), cpu_alloc);
    Tensor cpu_value_cache(DataTypeImpl::GetType<MLFloat16>(),
                           TensorShape({num_blocks, block_size, kv_num_heads, head_size}), cpu_alloc);
    ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(key_cache_value.Get<Tensor>(), cpu_key_cache));
    ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(value_cache_value.Get<Tensor>(), cpu_value_cache));
    const size_t cache_update_offset = static_cast<size_t>(past_seqlen * head_size);
    EXPECT_NEAR(cpu_key_cache.Data<MLFloat16>()[cache_update_offset].ToFloat(), 0.03f, 1e-3f);
    EXPECT_NEAR(cpu_value_cache.Data<MLFloat16>()[cache_update_offset].ToFloat(), 0.04f, 1e-3f);
    ASSERT_EQ(outputs.size(), 1u);
    return;
  }

  ASSERT_EQ(outputs.size(), 3u);
  if (alias_cache_outputs) {
    EXPECT_EQ(outputs[1].Get<Tensor>().Data<MLFloat16>(), key_cache_value.Get<Tensor>().Data<MLFloat16>());
    EXPECT_EQ(outputs[2].Get<Tensor>().Data<MLFloat16>(), value_cache_value.Get<Tensor>().Data<MLFloat16>());
  } else {
    EXPECT_NE(outputs[1].Get<Tensor>().Data<MLFloat16>(), key_cache_value.Get<Tensor>().Data<MLFloat16>());
    EXPECT_NE(outputs[2].Get<Tensor>().Data<MLFloat16>(), value_cache_value.Get<Tensor>().Data<MLFloat16>());
  }

  // Verify K/V scatter actually landed at slot `past_seqlen` in both caches.
  // The input caches are initialized to 0.01 / 0.02; the new K/V are 0.03 / 0.04.
  // Without this, a scatter regression on either path (alias or non-alias)
  // would still leave `output[0]` non-zero and pointer identity intact, so
  // the smoke-only version of this test could not distinguish "scatter ran"
  // from "scatter silently didn't run". Downloading from the bound output
  // tensors covers both the aliased path (output backed by the input cache
  // buffer) and the non-aliased path (output backed by a separate buffer).
  Tensor cpu_key_cache_out(DataTypeImpl::GetType<MLFloat16>(),
                           TensorShape({num_blocks, block_size, kv_num_heads, head_size}), cpu_alloc);
  Tensor cpu_value_cache_out(DataTypeImpl::GetType<MLFloat16>(),
                             TensorShape({num_blocks, block_size, kv_num_heads, head_size}), cpu_alloc);
  ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(outputs[1].Get<Tensor>(), cpu_key_cache_out));
  ORT_THROW_IF_ERROR(execution_provider_ptr->GetDataTransfer()->CopyTensor(outputs[2].Get<Tensor>(), cpu_value_cache_out));
  const size_t cache_update_offset = static_cast<size_t>(past_seqlen * head_size);
  EXPECT_NEAR(cpu_key_cache_out.Data<MLFloat16>()[cache_update_offset].ToFloat(), 0.03f, 1e-3f);
  EXPECT_NEAR(cpu_value_cache_out.Data<MLFloat16>()[cache_update_offset].ToFloat(), 0.04f, 1e-3f);
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

}  // namespace test
}  // namespace onnxruntime
