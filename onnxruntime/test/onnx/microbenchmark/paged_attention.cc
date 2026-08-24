// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Google Benchmark harness for the contrib PagedAttention op on the WebGPU EP.
//
// Sweeps B x {MHA, GQA} x head_size x prefill length. The direct-paged decode
// and fused paged-prefill programs are selected internally by
// PagedAttention::ComputeInternal / ApplyFlashAttention based on shape and
// adapter; there is no runtime toggle to fall back to gather-then-flash on
// shm-path adapters.
//
// Run:
//   onnxruntime_benchmark.exe --benchmark_filter=PagedAttention.* \
//                             --benchmark_min_time=0.5s

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/config_options.h"
#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tensor.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/session/IOBinding.h"
#include "core/session/ort_env.h"
#include "test/util/include/default_providers.h"

using onnxruntime::DataTypeImpl;
using onnxruntime::Graph;
using onnxruntime::IExecutionProvider;
using onnxruntime::InferenceSession;
using onnxruntime::IOBinding;
using onnxruntime::MLFloat16;
using onnxruntime::Model;
using onnxruntime::ModelMetaData;
using onnxruntime::ModelOptions;
using onnxruntime::NodeArg;
using onnxruntime::NodeAttributes;
using onnxruntime::PathString;
using onnxruntime::RunOptions;
using onnxruntime::SessionOptions;
using onnxruntime::Tensor;
using onnxruntime::TensorShape;

extern OrtEnv* env;

namespace {

struct PABenchCase {
  int batch_size;
  int num_heads;
  int kv_num_heads;
  int head_size;
  int seq_len;
  int past_seqlen;
  int block_size;
  // Optional per-batch Q lengths. Empty -> every batch has q_len == seq_len
  // (uniform); Setup() then derives token_count = batch_size * seq_len.
  // Set to test the varlen Q shader mode of the fused paged prefill kernel;
  // in that case seq_len is the *maximum* per-batch Q length and token_count
  // = sum(q_lens) (which can be less than batch_size * seq_len).
  std::vector<int> q_lens;
};

// Graph capture records the WebGPU command buffer once and replays it on each
// Run(), collapsing per-iteration CPU dispatch cost (~150-250us x N kernels on
// Windows/D3D12) to a single submit (~50us). Without graph capture this
// benchmark is CPU-dispatch-bound and the paged decode kernels' GPU-time
// advantage does not surface in wall time. Env-gated so the caller can A/B.
bool GraphCaptureEnabled() {
#ifdef _WIN32
  char buf[8];
  size_t len = 0;
  if (getenv_s(&len, buf, sizeof(buf), "ORT_WEBGPU_PA_BENCH_GRAPH_CAPTURE") != 0 || len == 0) {
    return false;
  }
  return len == 2 && buf[0] == '1';
#else
  const char* v = std::getenv("ORT_WEBGPU_PA_BENCH_GRAPH_CAPTURE");
  return v != nullptr && v[0] == '1' && v[1] == '\0';
#endif
}

struct PABenchContext {
  std::unique_ptr<Model> model;  // owns the Graph referenced by session
  std::unique_ptr<InferenceSession> session;
  std::unique_ptr<IOBinding> io_binding;
  RunOptions run_options;
  // Held so their backing tensors outlive io_binding.
  std::vector<OrtValue> bound_values;
};

// Builds a single-node PagedAttention model with the given shape, registers the
// WebGPU EP, uploads dummy fp16 inputs to the device, and returns a context
// whose Run() is directly measurable.
std::unique_ptr<PABenchContext> Setup(const PABenchCase& c) {
  const int batch = c.batch_size;
  const int T = c.seq_len;
  const int past = c.past_seqlen;
  const bool varlen = !c.q_lens.empty();
  // Sum of per-batch Q lengths (== batch*T in uniform mode).
  const int total_tokens = varlen
                               ? std::accumulate(c.q_lens.begin(), c.q_lens.end(), 0)
                               : batch * T;
  const int hidden_size = c.num_heads * c.head_size;
  const int kv_hidden_size = c.kv_num_heads * c.head_size;
  const int max_kv_len = past + T;
  const int max_num_blocks_per_seq = (max_kv_len + c.block_size - 1) / c.block_size + 1;
  const int num_blocks = max_num_blocks_per_seq * batch + 1;
  const int cache_elems = num_blocks * c.block_size * c.kv_num_heads * c.head_size;

  auto& logger = env->GetEnvironment().GetLoggingManager()->DefaultLogger();

  std::unordered_map<std::string, int> domain_to_version = {{onnxruntime::kMSDomain, 1}};
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  auto model = std::make_unique<Model>("paged_attention_bench", /*is_onnx_domain_only=*/true,
                                       ModelMetaData(), PathString(),
                                       onnxruntime::IOnnxRuntimeOpSchemaRegistryList(),
                                       domain_to_version, model_specific_functions, logger,
                                       ModelOptions(true, true));
  auto& graph = model->MainGraph();

  std::vector<ONNX_NAMESPACE::TypeProto> tensor_types;
  tensor_types.reserve(16);
  auto add_type = [&](int elem_type, std::initializer_list<int64_t> dims) -> ONNX_NAMESPACE::TypeProto* {
    tensor_types.emplace_back();
    auto* type = &tensor_types.back();
    type->mutable_tensor_type()->set_elem_type(elem_type);
    auto* shape = type->mutable_tensor_type()->mutable_shape();
    for (const int64_t dim : dims) {
      shape->add_dim()->set_dim_value(dim);
    }
    return type;
  };

  auto& query_arg = graph.GetOrCreateNodeArg(
      "query", add_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {total_tokens, hidden_size}));
  auto& key_arg = graph.GetOrCreateNodeArg(
      "key", add_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {total_tokens, kv_hidden_size}));
  auto& value_arg = graph.GetOrCreateNodeArg(
      "value", add_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {total_tokens, kv_hidden_size}));
  auto& key_cache_arg = graph.GetOrCreateNodeArg(
      "key_cache", add_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                            {num_blocks, c.block_size, c.kv_num_heads, c.head_size}));
  auto& value_cache_arg = graph.GetOrCreateNodeArg(
      "value_cache", add_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                              {num_blocks, c.block_size, c.kv_num_heads, c.head_size}));
  auto& cum_seqlens_arg = graph.GetOrCreateNodeArg(
      "cumulative_sequence_length", add_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch + 1}));
  auto& past_seqlens_arg = graph.GetOrCreateNodeArg(
      "past_seqlens", add_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch}));
  auto& block_table_arg = graph.GetOrCreateNodeArg(
      "block_table",
      add_type(ONNX_NAMESPACE::TensorProto_DataType_INT32, {batch, max_num_blocks_per_seq}));
  auto& empty_optional_arg = graph.GetOrCreateNodeArg("", nullptr);
  std::vector<NodeArg*> input_defs = {&query_arg, &key_arg, &value_arg, &key_cache_arg, &value_cache_arg,
                                      &cum_seqlens_arg, &past_seqlens_arg, &block_table_arg,
                                      &empty_optional_arg, &empty_optional_arg};

  auto& output_arg = graph.GetOrCreateNodeArg(
      "output", add_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {total_tokens, hidden_size}));
  auto& key_cache_out_arg = graph.GetOrCreateNodeArg(
      "key_cache_out", add_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                {num_blocks, c.block_size, c.kv_num_heads, c.head_size}));
  auto& value_cache_out_arg = graph.GetOrCreateNodeArg(
      "value_cache_out", add_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                                  {num_blocks, c.block_size, c.kv_num_heads, c.head_size}));
  std::vector<NodeArg*> output_defs = {&output_arg, &key_cache_out_arg, &value_cache_out_arg};

  NodeAttributes attrs = {
      {"num_heads", onnxruntime::utils::MakeAttribute("num_heads", int64_t{c.num_heads})},
      {"kv_num_heads", onnxruntime::utils::MakeAttribute("kv_num_heads", int64_t{c.kv_num_heads})},
      {"scale", onnxruntime::utils::MakeAttribute("scale", 0.0f)},
      {"do_rotary", onnxruntime::utils::MakeAttribute("do_rotary", int64_t{0})},
  };
  auto& node = graph.AddNode("paged_attention", "PagedAttention", "bench", input_defs, output_defs,
                             &attrs, onnxruntime::kMSDomain);
  node.SetExecutionProviderType(onnxruntime::kWebGpuExecutionProvider);
  if (!graph.Resolve().IsOK()) return nullptr;

  std::string model_string;
  if (!model->ToProto().SerializeToString(&model_string)) return nullptr;
  std::stringstream model_stream(model_string);

  auto ep_owned = [&]() -> std::unique_ptr<IExecutionProvider> {
    onnxruntime::ConfigOptions cfg{};
    ORT_THROW_IF_ERROR(cfg.AddConfigEntry(
        onnxruntime::webgpu::options::kStorageBufferCacheMode,
        onnxruntime::webgpu::options::kBufferCacheMode_Disabled));
    ORT_THROW_IF_ERROR(cfg.AddConfigEntry(
        onnxruntime::webgpu::options::kEnableInt64,
        onnxruntime::webgpu::options::kEnableInt64_ON));
    if (GraphCaptureEnabled()) {
      ORT_THROW_IF_ERROR(cfg.AddConfigEntry(
          onnxruntime::webgpu::options::kEnableGraphCapture,
          onnxruntime::webgpu::options::kEnableGraphCapture_ON));
    }
    return onnxruntime::test::WebGpuExecutionProviderWithOptions(cfg);
  }();
  if (ep_owned == nullptr) return nullptr;
  IExecutionProvider* ep = ep_owned.get();

  SessionOptions session_options;
  session_options.session_logid = "PagedAttentionBench";
  if (GraphCaptureEnabled()) {
    // Graph capture requires mem-pattern to be disabled (per graph_capture_test).
    session_options.enable_mem_pattern = false;
    // Mirror the EP-level graph capture flag on the session's config so the
    // WebGPU EP factory picks it up regardless of how it reads the option.
    ORT_THROW_IF_ERROR(session_options.config_options.AddConfigEntry(
        onnxruntime::webgpu::options::kEnableGraphCapture,
        onnxruntime::webgpu::options::kEnableGraphCapture_ON));
  }
  auto session = std::make_unique<InferenceSession>(session_options, env->GetEnvironment());
  if (!session->RegisterExecutionProvider(std::move(ep_owned)).IsOK()) return nullptr;

  auto device_allocators = ep->CreatePreferredAllocators();
  if (device_allocators.empty()) return nullptr;
  const OrtMemoryInfo* device_memory_info = nullptr;
  for (const auto& allocator : device_allocators) {
    const auto& mem_info = allocator->Info();
    if (mem_info.device.Type() == OrtDevice::GPU && mem_info.mem_type == OrtMemTypeDefault) {
      device_memory_info = &mem_info;
    }
  }
  if (device_memory_info == nullptr) return nullptr;
  const OrtMemoryInfo device_memory_info_copy = *device_memory_info;

  if (!session->Load(model_stream).IsOK() || !session->Initialize().IsOK()) return nullptr;

  auto device_alloc = session->GetAllocator(device_memory_info_copy);
  if (device_alloc == nullptr) return nullptr;
  auto cpu_alloc = onnxruntime::CPUAllocator::DefaultInstance();

  auto make_gpu_fp16 = [&](const std::vector<MLFloat16>& data, const TensorShape& shape) {
    Tensor cpu_tensor(DataTypeImpl::GetType<MLFloat16>(), shape,
                      const_cast<MLFloat16*>(data.data()), cpu_alloc->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<MLFloat16>(), shape, device_alloc);
    ORT_THROW_IF_ERROR(ep->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue value;
    Tensor::InitOrtValue(std::move(gpu_tensor), value);
    return value;
  };
  auto make_gpu_i32 = [&](const std::vector<int32_t>& data, const TensorShape& shape) {
    Tensor cpu_tensor(DataTypeImpl::GetType<int32_t>(), shape,
                      const_cast<int32_t*>(data.data()), cpu_alloc->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<int32_t>(), shape, device_alloc);
    ORT_THROW_IF_ERROR(ep->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue value;
    Tensor::InitOrtValue(std::move(gpu_tensor), value);
    return value;
  };

  std::vector<MLFloat16> query_data(total_tokens * hidden_size, MLFloat16(0.02f));
  std::vector<MLFloat16> kv_data(total_tokens * kv_hidden_size, MLFloat16(0.03f));
  std::vector<MLFloat16> cache_data(cache_elems, MLFloat16(0.01f));

  std::vector<int32_t> cum_seqlens(batch + 1);
  cum_seqlens[0] = 0;
  for (int b = 0; b < batch; ++b) {
    cum_seqlens[b + 1] = cum_seqlens[b] + (varlen ? c.q_lens[b] : T);
  }
  std::vector<int32_t> past_seqlens_vec(batch, past);
  std::vector<int32_t> block_table_vec(batch * max_num_blocks_per_seq);
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < max_num_blocks_per_seq; ++j) {
      block_table_vec[b * max_num_blocks_per_seq + j] = b * max_num_blocks_per_seq + j;
    }
  }

  auto ctx = std::make_unique<PABenchContext>();
  ctx->model = std::move(model);
  ctx->session = std::move(session);
  ctx->bound_values.reserve(11);

  auto push = [&](OrtValue v) -> OrtValue& {
    ctx->bound_values.push_back(std::move(v));
    return ctx->bound_values.back();
  };
  auto& query_v = push(make_gpu_fp16(query_data, TensorShape({total_tokens, hidden_size})));
  auto& key_v = push(make_gpu_fp16(kv_data, TensorShape({total_tokens, kv_hidden_size})));
  auto& value_v = push(make_gpu_fp16(kv_data, TensorShape({total_tokens, kv_hidden_size})));
  auto& key_cache_v = push(make_gpu_fp16(
      cache_data, TensorShape({num_blocks, c.block_size, c.kv_num_heads, c.head_size})));
  auto& value_cache_v = push(make_gpu_fp16(
      cache_data, TensorShape({num_blocks, c.block_size, c.kv_num_heads, c.head_size})));
  auto& cum_v = push(make_gpu_i32(cum_seqlens, TensorShape({batch + 1})));
  auto& past_v = push(make_gpu_i32(past_seqlens_vec, TensorShape({batch})));
  auto& block_table_v =
      push(make_gpu_i32(block_table_vec, TensorShape({batch, max_num_blocks_per_seq})));
  auto& output_v = push(make_gpu_fp16(std::vector<MLFloat16>(total_tokens * hidden_size),
                                      TensorShape({total_tokens, hidden_size})));

  if (!ctx->session->NewIOBinding(&ctx->io_binding).IsOK()) return nullptr;
  ORT_THROW_IF_ERROR(ctx->io_binding->BindInput("query", query_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindInput("key", key_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindInput("value", value_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindInput("key_cache", key_cache_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindInput("value_cache", value_cache_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindInput("cumulative_sequence_length", cum_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindInput("past_seqlens", past_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindInput("block_table", block_table_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindOutput("output", output_v));
  // Alias cache outputs to inputs so the kernel does not copy the whole cache
  // each iteration (matches the production paged-attention hot path).
  ORT_THROW_IF_ERROR(ctx->io_binding->BindOutput("key_cache_out", key_cache_v));
  ORT_THROW_IF_ERROR(ctx->io_binding->BindOutput("value_cache_out", value_cache_v));
  return ctx;
}

// Prefill benchmarks (T > 1).
void BM_PagedAttentionPrefill(benchmark::State& state) {
  PABenchCase c{
      static_cast<int>(state.range(0)),
      static_cast<int>(state.range(1)),
      static_cast<int>(state.range(2)),
      static_cast<int>(state.range(3)),
      static_cast<int>(state.range(4)),
      /*past_seqlen=*/0,
      /*block_size=*/256,
  };

  auto ctx = Setup(c);
  if (ctx == nullptr) {
    state.SkipWithError("PagedAttention bench setup failed (WebGPU EP unavailable?)");
    return;
  }

  // Warmup a handful of iterations before timing.
  for (int i = 0; i < 5; ++i) {
    auto st = ctx->session->Run(ctx->run_options, *ctx->io_binding);
    if (!st.IsOK()) {
      state.SkipWithError(st.ErrorMessage().c_str());
      return;
    }
  }
  auto sync_st = ctx->io_binding->SynchronizeOutputs();
  if (!sync_st.IsOK()) {
    state.SkipWithError(sync_st.ErrorMessage().c_str());
    return;
  }

  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    auto run_st = ctx->session->Run(ctx->run_options, *ctx->io_binding);
    if (!run_st.IsOK()) {
      state.SkipWithError(run_st.ErrorMessage().c_str());
      return;
    }
    auto out_sync = ctx->io_binding->SynchronizeOutputs();
    if (!out_sync.IsOK()) {
      state.SkipWithError(out_sync.ErrorMessage().c_str());
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }
}

// Variable-length prefill benchmark. Constructs a batch with per-batch Q
// lengths distributed as {T, T/2, T/4, T/8, ...} (halving) capped at
// batch_size entries. Exercises the varlen Q shader mode of the fused paged
// prefill kernel (mode b). Args:
// {batch, num_heads, kv_num_heads, head_size, max_T}
void BM_PagedAttentionPrefillVarlen(benchmark::State& state) {
  const int batch = static_cast<int>(state.range(0));
  const int max_T = static_cast<int>(state.range(4));
  std::vector<int> q_lens(batch);
  for (int b = 0; b < batch; ++b) {
    // Halving pattern, floored at 1: {max_T, max_T/2, max_T/4, ...}.
    const int q = std::max(1, max_T >> b);
    q_lens[b] = q;
  }
  PABenchCase c{
      batch,
      static_cast<int>(state.range(1)),
      static_cast<int>(state.range(2)),
      static_cast<int>(state.range(3)),
      max_T,
      /*past_seqlen=*/0,
      /*block_size=*/256,
      std::move(q_lens),
  };

  auto ctx = Setup(c);
  if (ctx == nullptr) {
    state.SkipWithError("PagedAttention varlen bench setup failed (WebGPU EP unavailable?)");
    return;
  }

  for (int i = 0; i < 5; ++i) {
    auto st = ctx->session->Run(ctx->run_options, *ctx->io_binding);
    if (!st.IsOK()) {
      state.SkipWithError(st.ErrorMessage().c_str());
      return;
    }
  }
  auto sync_st = ctx->io_binding->SynchronizeOutputs();
  if (!sync_st.IsOK()) {
    state.SkipWithError(sync_st.ErrorMessage().c_str());
    return;
  }

  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    auto run_st = ctx->session->Run(ctx->run_options, *ctx->io_binding);
    if (!run_st.IsOK()) {
      state.SkipWithError(run_st.ErrorMessage().c_str());
      return;
    }
    auto out_sync = ctx->io_binding->SynchronizeOutputs();
    if (!out_sync.IsOK()) {
      state.SkipWithError(out_sync.ErrorMessage().c_str());
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }
}

// Decode benchmarks (T == 1). Routed through the direct paged split-K decode
// kernels (FlashAttentionPagedDecodeQKV + FlashAttentionPagedDecodeVxReduce).
void BM_PagedAttentionDecode(benchmark::State& state) {
  PABenchCase c{
      static_cast<int>(state.range(0)),
      static_cast<int>(state.range(1)),
      static_cast<int>(state.range(2)),
      static_cast<int>(state.range(3)),
      /*seq_len=*/1,
      static_cast<int>(state.range(4)),
      /*block_size=*/256,
  };

  auto ctx = Setup(c);
  if (ctx == nullptr) {
    state.SkipWithError("PagedAttention bench setup failed (WebGPU EP unavailable?)");
    return;
  }

  for (int i = 0; i < 5; ++i) {
    auto st = ctx->session->Run(ctx->run_options, *ctx->io_binding);
    if (!st.IsOK()) {
      state.SkipWithError(st.ErrorMessage().c_str());
      return;
    }
  }
  auto sync_st = ctx->io_binding->SynchronizeOutputs();
  if (!sync_st.IsOK()) {
    state.SkipWithError(sync_st.ErrorMessage().c_str());
    return;
  }

  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    auto run_st = ctx->session->Run(ctx->run_options, *ctx->io_binding);
    if (!run_st.IsOK()) {
      state.SkipWithError(run_st.ErrorMessage().c_str());
      return;
    }
    auto out_sync = ctx->io_binding->SynchronizeOutputs();
    if (!out_sync.IsOK()) {
      state.SkipWithError(out_sync.ErrorMessage().c_str());
      return;
    }
    const auto end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// Registration: batch x head-config x seq_len.
//
// Head configs:
//   MHA_H64      : num_heads=16, kv_num_heads=16, head_size=64
//   MHA_H128     : num_heads=16, kv_num_heads=16, head_size=128
//   GQA_Qwen     : num_heads=14, kv_num_heads=2,  head_size=128  (ratio 7)
//   GQA_Llama    : num_heads=32, kv_num_heads=4,  head_size=128  (ratio 8)
//
// Prefill Ts:  {128, 512, 1024}
// Decode past: {512, 2048}
// -----------------------------------------------------------------------------

// Args: {batch, num_heads, kv_num_heads, head_size, seq_len}
#define REGISTER_PREFILL(BATCH, NH, NKV, H)    \
  BENCHMARK(BM_PagedAttentionPrefill)          \
      ->ArgNames({"B", "nH", "nKV", "H", "T"}) \
      ->Args({BATCH, NH, NKV, H, 128})         \
      ->Args({BATCH, NH, NKV, H, 512})         \
      ->Args({BATCH, NH, NKV, H, 1024})        \
      ->Unit(benchmark::kMicrosecond)          \
      ->UseManualTime()

REGISTER_PREFILL(1, 16, 16, 64);
REGISTER_PREFILL(1, 16, 16, 128);
REGISTER_PREFILL(1, 14, 2, 128);
REGISTER_PREFILL(1, 32, 4, 128);
REGISTER_PREFILL(2, 16, 16, 64);
REGISTER_PREFILL(2, 16, 16, 128);
REGISTER_PREFILL(2, 14, 2, 128);
REGISTER_PREFILL(2, 32, 4, 128);

#undef REGISTER_PREFILL

// Varlen prefill: q_lens = {max_T, max_T/2, max_T/4, ...} (halving pattern).
// Batch >= 2 required to be non-uniform. Args:
//   {batch, num_heads, kv_num_heads, head_size, max_T}
#define REGISTER_PREFILL_VARLEN(BATCH, NH, NKV, H) \
  BENCHMARK(BM_PagedAttentionPrefillVarlen)        \
      ->ArgNames({"B", "nH", "nKV", "H", "maxT"})  \
      ->Args({BATCH, NH, NKV, H, 512})             \
      ->Args({BATCH, NH, NKV, H, 1024})            \
      ->Unit(benchmark::kMicrosecond)              \
      ->UseManualTime()

REGISTER_PREFILL_VARLEN(2, 16, 16, 128);
REGISTER_PREFILL_VARLEN(2, 14, 2, 128);
REGISTER_PREFILL_VARLEN(2, 32, 4, 128);
REGISTER_PREFILL_VARLEN(4, 16, 16, 128);
REGISTER_PREFILL_VARLEN(4, 14, 2, 128);
REGISTER_PREFILL_VARLEN(4, 32, 4, 128);

#undef REGISTER_PREFILL_VARLEN

// Args: {batch, num_heads, kv_num_heads, head_size, past_seqlen}
#define REGISTER_DECODE(BATCH, NH, NKV, H)        \
  BENCHMARK(BM_PagedAttentionDecode)              \
      ->ArgNames({"B", "nH", "nKV", "H", "past"}) \
      ->Args({BATCH, NH, NKV, H, 512})            \
      ->Args({BATCH, NH, NKV, H, 2048})           \
      ->Unit(benchmark::kMicrosecond)             \
      ->UseManualTime()

REGISTER_DECODE(1, 16, 16, 64);
REGISTER_DECODE(1, 16, 16, 128);
REGISTER_DECODE(1, 14, 2, 128);
REGISTER_DECODE(1, 32, 4, 128);
REGISTER_DECODE(2, 16, 16, 64);
REGISTER_DECODE(2, 16, 16, 128);
REGISTER_DECODE(2, 14, 2, 128);
REGISTER_DECODE(2, 32, 4, 128);

#undef REGISTER_DECODE
