// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <cstring>
#include <limits>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/autoep/test_autoep_utils.h"
#include "test/autoep/library/example_plugin_ep/ep_test_hooks.h"
#include "test/util/include/stateful_execution_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime::test {
namespace {
struct Events {
  std::atomic<size_t> contexts{0};
  std::atomic<size_t> factories{0};
  std::atomic<size_t> release_logs{0};
};

void Observe(void* pointer, int kind) {
  auto& events = *static_cast<Events*>(pointer);
  if (kind == 1) ++events.contexts;
  if (kind == 2) ++events.factories;
}

void ORT_API_CALL Log(void* pointer, OrtLoggingLevel, const char*, const char*, const char*,
                      const char* message) {
  if (std::strstr(message, "Releasing retained stateful context")) {
    ++static_cast<Events*>(pointer)->release_logs;
  }
}

template <typename T>
Ort::Value Tensor(std::initializer_list<int64_t> shape, std::initializer_list<T> values) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto value = Ort::Value::CreateTensor<T>(allocator, shape.begin(), shape.size());
  if (values.size()) std::copy(values.begin(), values.end(), value.template GetTensorMutableData<T>());
  return value;
}

OrtStatefulProviderSelection Choice(OrtStatefulProviderMode mode = ORT_STATEFUL_PROVIDER_REQUIRE) {
  return {sizeof(OrtStatefulProviderSelection), mode, "example_ep", OrtMemoryInfoDeviceType_CPU, 0};
}

Ort::StatefulValue Output(const Ort::StatefulStepResult& result, const char* name) {
  for (const auto& value : result.GetOutputs()) {
    if (value.name == name) return value.value;
  }
  ORT_CXX_API_THROW(std::string("Missing output: ") + name, ORT_FAIL);
}

class StatefulPluginTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, registered_);
    ASSERT_NE(registered_, nullptr);
    Utils::LoadExampleEpHooks(Utils::example_ep_info, hooks_);
    ASSERT_NE(hooks_, nullptr);
    ASSERT_NE(hooks_->reset_stateful, nullptr);
    ASSERT_NE(hooks_->get_stateful_calls, nullptr);
    ASSERT_NE(hooks_->set_stateful_failure, nullptr);
    ASSERT_NE(hooks_->set_stateful_observer, nullptr);
    ASSERT_NE(hooks_->get_legacy_size, nullptr);
    hooks_->reset_stateful();
    hooks_->set_stateful_observer(&events_, &Observe);
  }
  void TearDown() override {
    registered_.reset();
    if (hooks_) hooks_->set_stateful_observer(nullptr, nullptr);
    hooks_.reset();
  }
  Ort::Session Session(const char* mode, ONNX_NAMESPACE::ModelProto model = stateful::MakeFeedbackModel()) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    options.SetLogSeverityLevel(0);
    Ort::ThrowOnError(Ort::GetApi().SetUserLoggingFunction(options, &Log, &events_));
    options.AddConfigEntry(kExampleEpTestStatefulMode, mode);
    options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
    options.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_.get())}, Ort::KeyValuePairs{});
    if (GetParam()) {
      Utils::AppendCudaExecutionProvider(*ort_env, options);
      options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    }
    const auto bytes = model.SerializeAsString();
    Ort::Session session(*ort_env, bytes.data(), bytes.size(), options);
    const auto assignments = session.GetEpGraphAssignmentInfo();
    EXPECT_FALSE(assignments.empty());
    for (const auto& graph : assignments) {
      EXPECT_EQ(graph.GetEpName(), GetParam() ? "CUDAExecutionProvider" : "CPUExecutionProvider");
    }
    return session;
  }
  Events events_;
  RegisteredEpDeviceUniquePtr registered_;
  Utils::LoadExampleEpHooksPtr hooks_;
};

TEST_P(StatefulPluginTest, IndependentModesAndGuardedLegacyAllocation) {
  for (const char* mode : {"none", "sampler", "append", "full", "extended", "legacy23"}) {
    hooks_->reset_stateful();
    auto session = Session(mode);
    const auto caps = Ort::QueryStatefulProviderCapabilities(session, "example_ep");
    const bool native_sampler = std::strcmp(mode, "sampler") == 0 || std::strcmp(mode, "full") == 0 ||
                                std::strcmp(mode, "extended") == 0;
    EXPECT_EQ((caps.operations & 64) != 0, native_sampler);
    if (std::strcmp(mode, "legacy23") == 0) {
      EXPECT_EQ(hooks_->get_legacy_size(), offsetof(OrtEp, GetKernelRegistry));
      EXPECT_LT(hooks_->get_legacy_size(), sizeof(OrtEp));
      EXPECT_EQ(caps.operations, 0u);
    }
    const OrtStatefulSamplerConfig config{sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
    Ort::SamplerOptions sampler(config);
    sampler.SetProvider(Choice(native_sampler ? ORT_STATEFUL_PROVIDER_REQUIRE : ORT_STATEFUL_PROVIDER_PREFER), session);
    auto logits = Tensor<float>({1, 2, 3}, {3.f, 1.f, 2.f, 1.f, 4.f, 4.f});
    EXPECT_EQ(sampler.Sample(logits), 1);
    const auto info = sampler.GetDispatchInfo();
    EXPECT_EQ(info.origin, native_sampler ? ORT_STATEFUL_DISPATCH_PROVIDER : ORT_STATEFUL_DISPATCH_CPU_FALLBACK);
    EXPECT_EQ(hooks_->get_stateful_calls(ORT_STATEFUL_SAMPLE_GREEDY), native_sampler ? 1u : 0u);
    if (!native_sampler) {
      EXPECT_THROW(sampler.SetProvider(Choice(), session), Ort::Exception);
    } else {
      auto half = Tensor<Ort::Float16_t>(
          {1, 3}, {Ort::Float16_t(-1.f), Ort::Float16_t(3.f), Ort::Float16_t(3.f)});
      EXPECT_EQ(sampler.Sample(half), 1);
      auto first_row = config;
      first_row.has_logits_position = 1;
      sampler.SetConfig(first_row).SetProvider(Choice(), session);
      EXPECT_EQ(sampler.Sample(logits), 0);
      auto nan = Tensor<float>({1, 2, 2}, {std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f, 2.f});
      EXPECT_THROW(sampler.Sample(nan), Ort::Exception);
      hooks_->set_stateful_failure(ORT_STATEFUL_SAMPLE_GREEDY, ORT_NOT_IMPLEMENTED);
      EXPECT_THROW(sampler.Sample(logits), Ort::Exception);
      EXPECT_EQ(sampler.GetDispatchInfo().cpu_calls, 0u);
      hooks_->set_stateful_failure(-1, 0);
    }
    if (std::strcmp(mode, "append") == 0) {
      const int64_t dimensions[]{1, 1, 0, 1};
      const OrtStatefulKVCacheSpec spec{"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dimensions, 4, 2, 0};
      const OrtStatefulKVCacheConfig caches{sizeof(caches), &spec, 1};
      Ort::KVCacheOptions options(caches);
      options.SetProvider(Choice(ORT_STATEFUL_PROVIDER_PREFER), session);
      Ort::AllocatorWithDefaultOptions allocator;
      Ort::KVCacheManager manager(options, allocator);
      auto delta = Tensor<float>({1, 1, 1, 1}, {3.f});
      manager.Update("key", delta, ORT_KV_CACHE_SHARE);
      manager.Update("key", delta, ORT_KV_CACHE_APPEND);
      EXPECT_EQ(manager.GetView("key").GetInfo().length, 2);
      EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_SHARE).origin, ORT_STATEFUL_DISPATCH_CPU_FALLBACK);
      EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_APPEND).provider_calls, 1u);
      EXPECT_EQ(hooks_->get_stateful_calls(ORT_STATEFUL_CACHE_APPEND), 1u);
    }
  }
}

TEST_P(StatefulPluginTest, RegistrationFailureRollbackAndOutsideScope) {
  EXPECT_EQ(offsetof(OrtEpApi, RegisterStatefulProvider),
            offsetof(OrtEpApi, EpContextConfigGetEpContextDataWriteFunc) + sizeof(void*));
  for (const char* mode : {"bad_version", "short", "missing_callback", "duplicate", "wrong_ep", "create_fail"}) {
    const auto before = events_.contexts.load();
    EXPECT_THROW(Session(mode), Ort::Exception) << mode;
    EXPECT_EQ(events_.contexts.load(), before + 1) << mode;
  }
  EXPECT_THROW(Session("unknown"), Ort::Exception);
  OrtEp ep{};
  OrtStatefulProviderDescriptor descriptor{};
  EXPECT_THROW(Ort::RegisterStatefulProvider(&ep, descriptor), Ort::Exception);
  auto* status = Ort::GetEpApi().RegisterStatefulProvider(nullptr, nullptr);
  ASSERT_NE(status, nullptr);
  Ort::GetApi().ReleaseStatus(status);
}

TEST_P(StatefulPluginTest, CacheErrorsRetainFactoryAllocatorAndLoggerAfterUnregister) {
  auto session = Session("full");
  const int64_t dimensions[4]{1, 1, 0, 1};
  const OrtStatefulKVCacheSpec spec{"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dimensions, 4, 2, 1};
  const OrtStatefulKVCacheConfig config{sizeof(config), &spec, 1};
  Ort::KVCacheOptions options(config);
  options.SetProvider(Choice(), session);
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::KVCacheManager manager(options, allocator);
  auto full = Tensor<float>({1, 1, 1, 1}, {3.f});
  manager.Update("key", full, ORT_KV_CACHE_SHARE);
  auto delta = Tensor<float>({1, 1, 1, 1}, {4.f});
  manager.Update("key", delta, ORT_KV_CACHE_APPEND);
  manager.Reserve("key", 6);
  hooks_->set_stateful_failure(ORT_STATEFUL_CACHE_APPEND, ORT_NOT_IMPLEMENTED);
  EXPECT_THROW(manager.Update("key", delta, ORT_KV_CACHE_APPEND), Ort::Exception);
  EXPECT_TRUE(manager.RequiresReset());
  EXPECT_EQ(manager.GetDispatchInfo(ORT_STATEFUL_CACHE_APPEND).cpu_calls, 0u);
  EXPECT_THROW(manager.GetView("key"), Ort::Exception);
  hooks_->set_stateful_failure(ORT_STATEFUL_CACHE_RESET, ORT_EP_FAIL);
  EXPECT_THROW(manager.Reset(), Ort::Exception);
  EXPECT_TRUE(manager.RequiresReset());
  hooks_->set_stateful_failure(-1, 0);
  manager.Reset();
  manager.Update("key", full, ORT_KV_CACHE_COPY);
  auto retained = manager.GetView("key").GetInfo();
  const auto diagnostic = manager.GetDispatchInfo(ORT_STATEFUL_CACHE_COPY);
  EXPECT_GT(hooks_->get_stateful_calls(ORT_STATEFUL_CACHE_COPY), 0u);
  manager = Ort::KVCacheManager(nullptr);
  options = Ort::KVCacheOptions(nullptr);
  session = Ort::Session(nullptr);
  EXPECT_EQ(events_.contexts.load(), 0u);
  Ort::ThrowOnError(Ort::GetApi().UnregisterExecutionProviderLibrary(*ort_env, "example_ep"));
  registered_.release();
  EXPECT_EQ(events_.factories.load(), 0u);
  EXPECT_EQ(retained.value.GetTensorData<float>()[0], 3.f);
  retained.value = Ort::StatefulValue(nullptr);
  EXPECT_EQ(events_.contexts.load(), 1u);
  EXPECT_EQ(events_.release_logs.load(), 1u);
  EXPECT_EQ(events_.factories.load(), 1u);
  EXPECT_EQ(diagnostic.provider_name, "example_ep");
}

ONNX_NAMESPACE::ModelProto PluginGenerationModel() {
  auto model = stateful::MakeDecoderModel();
  auto& graph = *model.mutable_graph();
  graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  graph.mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  stateful::SetTensorType(*graph.mutable_input(0), "token", ONNX_NAMESPACE::TensorProto_DataType_INT64, {1, -1});
  stateful::SetTensorType(*graph.mutable_output(0), "logits", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  for (auto& initializer : *graph.mutable_initializer()) {
    if (initializer.name() == "key_shape") {
      initializer.clear_int64_data();
      for (int64_t value : {1, 1, -1, 1}) initializer.add_int64_data(value);
    }
  }
  return model;
}

TEST_P(StatefulPluginTest, PluginPrefillDecodeAndRetainedResults) {
  auto session = Session("full", PluginGenerationModel());
  const int64_t dimensions[4]{1, 1, 0, 1};
  const OrtStatefulKVCacheBinding cache{
      {"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dimensions, 4, 2, 0},
      "past_key",
      "present_key",
      ORT_KV_CACHE_SHARE};
  const char* finite[1]{"token"};
  const char* outputs[1]{"logits"};
  const OrtStatefulExitCondition prefill_exits[2]{
      {ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0},
      {ORT_STATEFUL_MAX_GENERATED_TOKENS, 2, nullptr, nullptr, 0}};
  const OrtStatefulExitCondition decode_exit{ORT_STATEFUL_MAX_GENERATED_TOKENS, 2, nullptr, nullptr, 0};
  OrtStatefulGenerationConfig generation{
      sizeof(generation), ORT_STATEFUL_PREFILL, "token", "logits", nullptr, nullptr, 0, 0};
  OrtStatefulLoopConfig config{};
  config.struct_size = sizeof(config);
  config.finite_input_names = finite;
  config.num_finite_inputs = 1;
  config.output_names = outputs;
  config.num_outputs = 1;
  config.exit_conditions = prefill_exits;
  config.num_exit_conditions = 2;
  config.kv_caches = &cache;
  config.num_kv_caches = 1;
  config.generation = &generation;
  Ort::SessionLoopOptions prefill_options(config);
  prefill_options.SetProvider(ORT_STATEFUL_FEATURE_CACHE, Choice());
  prefill_options.SetProvider(ORT_STATEFUL_FEATURE_SAMPLER, Choice());
  Ort::SessionLoop prefill(session, prefill_options);
  generation.phase = ORT_STATEFUL_DECODE;
  config.num_finite_inputs = 0;
  config.exit_conditions = &decode_exit;
  config.num_exit_conditions = 1;
  Ort::SessionLoopOptions decode_options(config);
  decode_options.SetProvider(ORT_STATEFUL_FEATURE_CACHE, Choice());
  decode_options.SetProvider(ORT_STATEFUL_FEATURE_SAMPLER, Choice());
  Ort::SessionLoop decode(session, decode_options);
  const OrtStatefulInputBinding binding{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
  OrtStatefulChainComponent components[2]{};
  components[0].name = "prefill";
  components[0].loop = prefill;
  components[0].inputs = &binding;
  components[0].num_inputs = 1;
  components[1].name = "decode";
  components[1].loop = decode;
  const OrtStatefulChainOutput output{"tokens", 0, nullptr, ORT_STATEFUL_GENERATED_TOKEN_IDS};
  const OrtStatefulChainConfig chain_config{sizeof(chain_config), components, 2, &output, 1};
  Ort::SessionChain chain{Ort::SessionChainOptions(chain_config)};
  auto prompt = Tensor<int64_t>({1, 1}, {0});
  const OrtStatefulNamedValue input{"prompt", prompt};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &input;
  inputs.num_inputs = 1;
  auto execution = chain.CreateExecution(inputs);
  auto result = execution.Run();
  auto tokens = Output(result, "tokens");
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[0], 1);
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[1], 2);
  EXPECT_EQ(hooks_->get_stateful_calls(ORT_STATEFUL_SAMPLE_GREEDY), 2u);
  EXPECT_EQ(hooks_->get_stateful_calls(ORT_STATEFUL_CACHE_SHARE), 2u);
  EXPECT_EQ(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 1).provider_calls, 1u);
  execution.Reset();
  result = Ort::StatefulStepResult(nullptr);
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[1], 2);
}

INSTANTIATE_TEST_SUITE_P(Cpu, StatefulPluginTest, testing::Values(false));
#ifdef USE_CUDA
INSTANTIATE_TEST_SUITE_P(Cuda, StatefulPluginTest, testing::Values(true));
#endif
}  // namespace
}  // namespace onnxruntime::test
