// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>

#include <cuda_runtime_api.h>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/stateful_execution_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" int StatefulApiCGpuLoopIntegrationTest(OrtSession*, const OrtValue*, OrtStatefulStepResult**);

namespace onnxruntime::test {
namespace {

OrtStatefulProviderSelection GpuCache(int id = 0) {
  return {sizeof(OrtStatefulProviderSelection), ORT_STATEFUL_PROVIDER_REQUIRE, "CUDAExecutionProvider",
          OrtMemoryInfoDeviceType_GPU, id};
}

Ort::StatefulValue Named(const Ort::StatefulStepResult& result, const char* name,
                         OrtStatefulOutputGroup group = ORT_STATEFUL_SELECTED_OUTPUTS) {
  for (const auto& item : result.GetOutputs(group)) {
    if (item.name == name) return item.value;
  }
  ORT_CXX_API_THROW(std::string("Missing stateful output: ") + name, ORT_FAIL);
}

class CudaCacheApiIntegrationTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(cudaGetDevice(&previous_device_), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
  void TearDown() override { EXPECT_EQ(cudaSetDevice(previous_device_), cudaSuccess); }

  Ort::Session Session(bool gpu = true, cudaStream_t stream = nullptr,
                       ONNX_NAMESPACE::ModelProto model = stateful::MakeDynamicDecoderModel(), int id = 0) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1).SetInterOpNumThreads(1).SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
    if (gpu) {
      Ort::ConstEpDevice selected(nullptr);
      for (auto device : ort_env->GetEpDevices()) {
        if (std::strcmp(device.EpName(), "CUDAExecutionProvider") == 0 &&
            device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT).GetDeviceId() == id) selected = device;
      }
      if (!selected) ORT_CXX_API_THROW("Required CUDA plugin test device is unavailable", ORT_FAIL);
      if (stream) {
        options.AddConfigEntry("ep.cuda.user_compute_stream",
                               std::to_string(reinterpret_cast<uintptr_t>(stream)).c_str());
      }
      options.AppendExecutionProvider_V2(*ort_env, {selected}, Ort::KeyValuePairs{});
      options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    }
    const auto data = model.SerializeAsString();
    Ort::Session session(*ort_env, data.data(), data.size(), options);
    for (const auto& graph : session.GetEpGraphAssignmentInfo()) {
      EXPECT_EQ(graph.GetEpName(), gpu ? "CUDAExecutionProvider" : "CPUExecutionProvider");
    }
    return session;
  }

  Ort::Value Tokens(std::initializer_list<int64_t> tokens = {0, 1}) {
    Ort::AllocatorWithDefaultOptions allocator;
    const int64_t shape[]{1, static_cast<int64_t>(tokens.size())};
    auto value = Ort::Value::CreateTensor<int64_t>(allocator, shape, 2);
    std::copy(tokens.begin(), tokens.end(), value.GetTensorMutableData<int64_t>());
    return value;
  }

  static OrtStatefulKVCacheBinding CacheBinding() {
    static const int64_t shape[]{1, 1, 0, 1};
    return {{"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 4, 2, 0},
            "past_key",
            "present_key",
            ORT_KV_CACHE_SHARE};
  }

  Ort::SessionLoop Loop(Ort::Session& session, bool gpu = true, int phase = 0, int64_t budget = 3,
                        bool eos = false) {
    const char* finite[]{"token"};
    const char* outputs[]{"logits", "present_key"};
    const int64_t eos_id = 3;
    const std::array<OrtStatefulExitCondition, 3> exits{{{ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0},
                                                         {ORT_STATEFUL_MAX_GENERATED_TOKENS, budget,
                                                          nullptr, nullptr, 0},
                                                         {ORT_STATEFUL_EOS, 0, nullptr, &eos_id, 1}}};
    const auto cache = CacheBinding();
    const OrtStatefulGenerationConfig generation{
        sizeof(generation), phase == 1 ? ORT_STATEFUL_PREFILL : ORT_STATEFUL_DECODE,
        "token", "logits", nullptr, nullptr, 0, 0};
    OrtStatefulLoopConfig config{};
    config.struct_size = sizeof(config);
    config.finite_input_names = phase == 2 ? nullptr : finite;
    config.num_finite_inputs = phase == 2 ? 0 : 1;
    config.output_names = outputs;
    config.num_outputs = 2;
    config.exit_conditions = phase == 2 ? &exits[1] : exits.data();
    config.num_exit_conditions = phase == 0 ? 1 : phase == 2 ? (eos ? 2 : 1)
                                                             : (eos ? 3 : 2);
    config.kv_caches = &cache;
    config.num_kv_caches = 1;
    config.generation = phase ? &generation : nullptr;
    Ort::SessionLoopOptions options(config);
    if (gpu) options.SetProvider(ORT_STATEFUL_FEATURE_CACHE, GpuCache());
    return Ort::SessionLoop(session, options);
  }

  Ort::SessionChain GenerationChain(Ort::Session& prefill, Ort::Session& decode, bool gpu, int64_t budget,
                                    bool eos = false) {
    auto first = Loop(prefill, gpu, 1, budget, eos), second = Loop(decode, gpu, 2, budget, eos);
    const OrtStatefulInputBinding prompt{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
    const OrtStatefulChainComponent stages[]{
        {"prefill", nullptr, first, &prompt, 1, nullptr, 0, nullptr, 0},
        {"decode", nullptr, second, nullptr, 0, nullptr, 0, nullptr, 0}};
    const OrtStatefulChainOutput output{"tokens", 0, nullptr, ORT_STATEFUL_GENERATED_TOKEN_IDS};
    const OrtStatefulChainConfig config{sizeof(config), stages, 2, &output, 1};
    return Ort::SessionChain(Ort::SessionChainOptions(config));
  }

  static std::vector<float> Read(const Ort::StatefulValue& value) {
    std::vector<float> data(value.GetTensorTypeAndShapeInfo().GetElementCount());
    if (!data.empty()) {
      if (value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU) {
        EXPECT_EQ(cudaMemcpy(data.data(), value.GetTensorRawData(), data.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
      } else {
        std::copy_n(value.GetTensorData<float>(), data.size(), data.data());
      }
    }
    return data;
  }
  int previous_device_ = 0;
};

TEST_F(CudaCacheApiIntegrationTest, PrefillDecodeBudgetEosCpuParityAndResidentState) {
  cudaStream_t a = nullptr, b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
  {
    auto prefill = Session(true, a), decode = Session(true, b);
    auto cpu_prefill = Session(false), cpu_decode = Session(false);
    auto prompt = Tokens();
    const OrtStatefulNamedValue feed{"prompt", prompt};
    OrtStatefulInputs inputs{};
    inputs.struct_size = sizeof(inputs);
    inputs.inputs = &feed;
    inputs.num_inputs = 1;
    for (const int64_t budget : {0, 1, 3}) {
      for (const bool eos : {false, true}) {
        auto chain = GenerationChain(prefill, decode, true, budget, eos);
        auto reference = GenerationChain(cpu_prefill, cpu_decode, false, budget, eos);
        auto execution = chain.CreateExecution(inputs), cpu = reference.CreateExecution(inputs);
        auto independent = chain.CreateExecution(inputs);
        std::vector<Ort::StatefulStepResult> retained;
        Ort::StatefulStepResult result(nullptr);
        do {
          result = execution.Step();
          ASSERT_EQ(cudaStreamQuery(a), cudaSuccess);
          ASSERT_EQ(cudaStreamQuery(b), cudaSuccess);
          if (result.GetInfo().executed) {
            const auto cache = Named(result, "present_key", ORT_STATEFUL_COMPONENT_OUTPUTS);
            EXPECT_EQ(cache.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
            EXPECT_EQ(Named(result, "logits", ORT_STATEFUL_COMPONENT_OUTPUTS)
                          .GetTensorMemoryInfo()
                          .GetDeviceType(),
                      OrtMemoryInfoDeviceType_CPU);
            retained.push_back(result);
          }
        } while (!result.GetInfo().completed);
        auto expected = cpu.Run();
        const auto actual_tokens = Named(result, "tokens");
        const auto expected_tokens = Named(expected, "tokens");
        EXPECT_EQ(actual_tokens.GetTensorTypeAndShapeInfo().GetShape(),
                  expected_tokens.GetTensorTypeAndShapeInfo().GetShape());
        const size_t count = actual_tokens.GetTensorTypeAndShapeInfo().GetElementCount();
        if (count) {
          EXPECT_TRUE(std::equal(actual_tokens.GetTensorData<int64_t>(),
                                 actual_tokens.GetTensorData<int64_t>() + count,
                                 expected_tokens.GetTensorData<int64_t>()));
        }
        auto state = execution.GetState();
        const auto value = Named(state, "key", ORT_STATEFUL_STATE_VALUES);
        EXPECT_EQ(value.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
        EXPECT_EQ(Read(value), Read(Named(cpu.GetState(), "key", ORT_STATEFUL_STATE_VALUES)));
        EXPECT_TRUE(Read(Named(independent.GetState(), "key", ORT_STATEFUL_STATE_VALUES)).empty());
        if (budget) {
          const auto dispatch = execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_CACHE, ORT_STATEFUL_CACHE_SHARE);
          EXPECT_GT(dispatch.provider_calls, 0u);
          EXPECT_EQ(dispatch.cpu_calls, 0u);
          EXPECT_EQ(dispatch.cpu_delegations, 0u);
          const auto sampler = execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY);
          EXPECT_GT(sampler.cpu_calls, 0u);
          EXPECT_EQ(sampler.provider_calls, 0u);
          EXPECT_EQ(execution.GetStagingInfo(0).cpu_output_bytes, 32u);
          EXPECT_EQ(execution.GetStagingInfo(0).cpu_snapshot_copy_bytes, 32u);
        }
        const auto saved = Read(value);
        execution.Reset();
        execution.Run();
        EXPECT_EQ(Read(value), saved);
        if (!retained.empty()) EXPECT_EQ(Read(Named(retained.front(), "present_key",
                                                    ORT_STATEFUL_COMPONENT_OUTPUTS)),
                                         (std::vector<float>{0, 1}));
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(b), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(a), cudaSuccess);
}

TEST_F(CudaCacheApiIntegrationTest, AllSessionLoopChainFormsRetainUnselectedGpuOutputs) {
  auto first = Session(), second = Session();
  auto first_loop = Loop(first), second_loop = Loop(second);
  auto prompt = Tokens(), next = Tokens({2});
  const auto cache = CacheBinding();
  const OrtStatefulInputBinding initial{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
  const OrtStatefulInputBinding subsequent{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "next", 0}};
  const OrtStatefulNamedValue feeds[]{{"prompt", prompt}, {"next", next}};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = feeds;
  inputs.num_inputs = 2;
  for (const bool a : {false, true}) {
    for (const bool b : {false, true}) {
      const OrtStatefulChainComponent components[]{
          {"first", a ? nullptr : first, a ? first_loop : nullptr, &initial, 1, nullptr, 0,
           a ? nullptr : &cache, a ? 0u : 1u},
          {"second", b ? nullptr : second, b ? second_loop : nullptr, &subsequent, 1, nullptr, 0,
           b ? nullptr : &cache, b ? 0u : 1u}};
      const OrtStatefulChainOutput outputs[]{
          {"first_key", 0, "present_key", ORT_STATEFUL_MODEL_OUTPUT},
          {"last_key", 1, "present_key", ORT_STATEFUL_MODEL_OUTPUT},
          {"logits", 1, "logits", ORT_STATEFUL_MODEL_OUTPUT}};
      const OrtStatefulChainConfig config{sizeof(config), components, 2, outputs, 3};
      Ort::SessionChainOptions options(config);
      if (!a) options.SetComponentProvider(0, ORT_STATEFUL_FEATURE_CACHE, GpuCache());
      if (!b) options.SetComponentProvider(1, ORT_STATEFUL_FEATURE_CACHE, GpuCache());
      Ort::SessionChain chain(options);
      auto execution = chain.CreateExecution(inputs);
      auto result = execution.Run();
      EXPECT_EQ(Read(Named(result, "first_key")), (std::vector<float>{0, 1}));
      EXPECT_EQ(Read(Named(result, "last_key")), (std::vector<float>{0, 1, 2}));
      EXPECT_EQ(Named(result, "last_key").GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
      EXPECT_EQ(execution.GetStagingInfo(0).cpu_snapshot_copy_bytes, 32u);
      EXPECT_EQ(execution.GetStagingInfo(1).cpu_snapshot_copy_bytes, 16u);
    }
  }
}

TEST_F(CudaCacheApiIntegrationTest, CConsumerResultsSurviveAllExecutionAndSessionOwners) {
  auto session = Session();
  auto tokens = Tokens();
  OrtStatefulStepResult* raw = nullptr;
  ASSERT_EQ(StatefulApiCGpuLoopIntegrationTest(session, tokens, &raw), 0);
  Ort::StatefulStepResult result(raw);
  auto cache = Named(result, "present_key");
  result = Ort::StatefulStepResult(nullptr);
  session = Ort::Session(nullptr);
  EXPECT_EQ(cache.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  EXPECT_EQ(Read(cache), (std::vector<float>{0, 1}));
  ort_env->UnregisterExecutionProviderLibrary("CUDAExecutionProvider");
  EXPECT_EQ(Read(cache), (std::vector<float>{0, 1}));
  cache = Ort::StatefulValue(nullptr);
  ort_env->RegisterExecutionProviderLibrary(
      "CUDAExecutionProvider", std::filesystem::path(ORT_UNIT_TEST_CUDA_PLUGIN_EP_LIBRARY_PATH).native());
}

TEST_F(CudaCacheApiIntegrationTest, LeadingCpuStageGpuComponentAndNamedStateCacheAliases) {
  auto seed_model = stateful::MakeModel("leading_cpu");
  auto& seed_graph = *seed_model.mutable_graph();
  auto* seed = seed_graph.add_initializer();
  seed->set_name("seed");
  seed->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  seed->add_float_data(1.f);
  stateful::SetTensorType(*seed_graph.add_output(), "seed_output", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  stateful::AddNode(seed_graph, "Identity", {"seed"}, "seed_output");
  auto identity_model = stateful::MakeModel("cache_alias");
  auto& identity = *identity_model.mutable_graph();
  stateful::SetTensorType(*identity.add_input(), "incoming", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, -1, 1});
  stateful::SetTensorType(
      *identity.add_output(), "forwarded", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, -1, 1});
  stateful::AddNode(identity, "Identity", {"incoming"}, "forwarded");
  auto sum_model = stateful::MakeModel("cache_consumer");
  auto& sum = *sum_model.mutable_graph();
  stateful::SetTensorType(*sum.add_input(), "incoming", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, -1, 1});
  stateful::SetTensorType(*sum.add_output(), "summary", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  auto& reduce = stateful::AddNode(sum, "ReduceSum", {"incoming"}, "summary");
  auto* keepdims = reduce.add_attribute();
  keepdims->set_name("keepdims");
  keepdims->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  keepdims->set_i(0);
  auto leading = Session(false, nullptr, seed_model);
  auto producer = Session();
  auto reader = Session(true, nullptr, identity_model);
  auto consumer = Session(true, nullptr, sum_model);
  auto wrong_reader = Session(false, nullptr, identity_model);
  auto prompt = Tokens();
  const OrtStatefulNamedValue feed{"prompt", prompt};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &feed;
  inputs.num_inputs = 1;
  const auto cache = CacheBinding();
  const OrtStatefulInputBinding prompt_binding{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
  const OrtStatefulStateBinding aliases[]{{"alias", "present_key"}, {"ordinary", "logits"}};
  const OrtStatefulStateBinding write_cache{"key", "forwarded"};
  const OrtStatefulInputBinding consume{"incoming", {ORT_STATEFUL_STATE, "key", 0}};
  const char* finite[]{"incoming"};
  const char* selected[]{"forwarded"};
  const OrtStatefulExitCondition end{ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0};
  OrtStatefulLoopConfig reader_config{};
  reader_config.struct_size = sizeof(reader_config);
  reader_config.finite_input_names = finite;
  reader_config.num_finite_inputs = 1;
  reader_config.output_names = selected;
  reader_config.num_outputs = 1;
  reader_config.exit_conditions = &end;
  reader_config.num_exit_conditions = 1;
  Ort::SessionLoop reader_loop(reader, Ort::SessionLoopOptions(reader_config));
  for (const bool from_state : {false, true}) {
    const auto kind = from_state ? ORT_STATEFUL_STATE : ORT_STATEFUL_COMPONENT_OUTPUT;
    const OrtStatefulInputBinding route{"incoming", {kind, from_state ? "alias" : "present_key", 1}};
    OrtStatefulChainComponent stages[]{
        {"leading", leading, nullptr, nullptr, 0, nullptr, 0, nullptr, 0},
        {"producer", producer, nullptr, &prompt_binding, 1, aliases, 2, &cache, 1},
        {"reader", from_state ? nullptr : reader, from_state ? reader_loop : nullptr,
         &route, 1, &write_cache, 1, nullptr, 0},
        {"consumer", consumer, nullptr, &consume, 1, nullptr, 0, nullptr, 0}};
    const OrtStatefulChainOutput outputs[]{
        {"cache", 2, "forwarded", ORT_STATEFUL_MODEL_OUTPUT},
        {"sum", 3, "summary", ORT_STATEFUL_MODEL_OUTPUT}};
    const OrtStatefulChainConfig config{sizeof(config), stages, 4, outputs, 2};
    Ort::SessionChainOptions options(config);
    options.SetComponentProvider(1, ORT_STATEFUL_FEATURE_CACHE, GpuCache());
    Ort::SessionChain chain(options);
    auto execution = chain.CreateExecution(inputs);
    auto result = execution.Run();
    auto state = execution.GetState();
    EXPECT_EQ(Read(Named(result, "cache")), (std::vector<float>{0, 1}));
    EXPECT_EQ(Named(result, "cache").GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
    EXPECT_EQ(Named(result, "sum").GetTensorData<float>()[0], 1.f);
    EXPECT_EQ(Named(state, "alias", ORT_STATEFUL_STATE_VALUES).GetTensorMemoryInfo().GetDeviceType(),
              OrtMemoryInfoDeviceType_GPU);
    EXPECT_EQ(execution.GetStagingInfo(2).cpu_input_request_bytes, 0u);
    EXPECT_EQ(execution.GetStagingInfo(2).cpu_output_bytes, 0u);
    EXPECT_EQ(execution.GetStagingInfo(2).cpu_snapshot_copy_bytes, 0u);
    EXPECT_EQ(execution.GetStagingInfo(3).cpu_input_request_bytes, 0u);
    EXPECT_EQ(execution.GetStagingInfo(3).cpu_snapshot_copy_bytes, 4u);
    stages[2].session = wrong_reader;
    stages[2].loop = nullptr;
    Ort::SessionChainOptions invalid(config);
    invalid.SetComponentProvider(1, ORT_STATEFUL_FEATURE_CACHE, GpuCache());
    EXPECT_THROW(Ort::SessionChain{invalid}, Ort::Exception);
  }
}

TEST_F(CudaCacheApiIntegrationTest, SharedCachesRejectConflictingDevicesBeforeExecution) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  auto first = Session();
  auto second = Session(true, nullptr, stateful::MakeDynamicDecoderModel(), count > 1 ? 1 : 0);
  const auto cache = CacheBinding();
  const OrtStatefulInputBinding input{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
  const OrtStatefulChainComponent stages[]{
      {"first", first, nullptr, &input, 1, nullptr, 0, &cache, 1},
      {"second", second, nullptr, &input, 1, nullptr, 0, &cache, 1}};
  const OrtStatefulChainOutput output{"cache", 1, "present_key", ORT_STATEFUL_MODEL_OUTPUT};
  const OrtStatefulChainConfig config{sizeof(config), stages, 2, &output, 1};
  Ort::SessionChainOptions options(config);
  options.SetComponentProvider(0, ORT_STATEFUL_FEATURE_CACHE, GpuCache());
  if (count > 1) {
    options.SetComponentProvider(1, ORT_STATEFUL_FEATURE_CACHE, GpuCache(1));
  }
  EXPECT_THROW(Ort::SessionChain{options}, Ort::Exception);
  RecordProperty("cross_device_cache_tested", count > 1);
}

}  // namespace
}  // namespace onnxruntime::test

#endif
