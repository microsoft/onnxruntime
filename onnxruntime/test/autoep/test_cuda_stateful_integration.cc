// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

#include <cuda_runtime_api.h>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/stateful_execution_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" int StatefulApiCGpuLoopIntegrationTest(OrtSession*, const OrtValue*, int, OrtStatefulStepResult**);

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

  static ONNX_NAMESPACE::ModelProto SamplingModel(bool half = false, int64_t vocabulary = 4) {
    auto model = stateful::MakeDynamicDecoderModel();
    auto& graph = *model.mutable_graph();
    graph.mutable_output(0)->mutable_type()->mutable_tensor_type()->clear_shape();
    stateful::SetTensorType(*graph.mutable_output(0), "logits",
                            half ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                                 : ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                            {1, -1, vocabulary});
    auto* scores = graph.mutable_initializer(0);
    scores->set_dims(1, vocabulary);
    scores->clear_float_data();
    for (int64_t token = 0; token < 4; ++token) {
      for (int64_t id = 0; id < vocabulary; ++id) {
        scores->add_float_data(id == (token + 1) % 4 ? 8.f : -1.f);
      }
    }
    if (half) {
      graph.mutable_node(0)->set_output(0, "float_logits");
      auto& cast = stateful::AddNode(graph, "Cast", {"float_logits"}, "logits");
      auto* to = cast.add_attribute();
      to->set_name("to");
      to->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
      to->set_i(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    }
    return model;
  }

  Ort::SessionLoop Loop(Ort::Session& session, bool gpu = true, int phase = 0, int64_t budget = 3,
                        bool eos = false, bool gpu_sampler = false,
                        std::optional<int64_t> position = std::nullopt) {
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
        "token", "logits", nullptr, nullptr, position.has_value(), position.value_or(0)};
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
    if (gpu_sampler) options.SetProvider(ORT_STATEFUL_FEATURE_SAMPLER, GpuCache());
    return Ort::SessionLoop(session, options);
  }

  Ort::SessionChain GenerationChain(Ort::Session& prefill, Ort::Session& decode, bool gpu, int64_t budget,
                                    bool eos = false, bool prefill_gpu_sampler = false,
                                    bool decode_gpu_sampler = false) {
    auto first = Loop(prefill, gpu, 1, budget, eos, prefill_gpu_sampler);
    auto second = Loop(decode, gpu, 2, budget, eos, decode_gpu_sampler);
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
  static std::vector<uint8_t> ReadBytes(const Ort::StatefulValue& value) {
    std::vector<uint8_t> bytes(value.GetTensorSizeInBytes());
    if (!bytes.empty()) {
      if (value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU) {
        const auto status = cudaMemcpy(bytes.data(), value.GetTensorRawData(), bytes.size(), cudaMemcpyDeviceToHost);
        EXPECT_EQ(status, cudaSuccess);
      } else {
        std::memcpy(bytes.data(), value.GetTensorRawData(), bytes.size());
      }
    }
    return bytes;
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
  ASSERT_EQ(StatefulApiCGpuLoopIntegrationTest(session, tokens, 0, &raw), 0);
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

class CudaSamplingApiIntegrationTest : public CudaCacheApiIntegrationTest {};

TEST_F(CudaSamplingApiIntegrationTest, GenerationMatchesCpuForIndependentStagePoliciesAndDtypes) {
  cudaStream_t a = nullptr, b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
  auto release = [](cudaStream_t stream) { EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess); };
  using StreamOwner = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(release)>;
  StreamOwner first_stream(a, release), second_stream(b, release);
  auto prompt = Tokens();
  const OrtStatefulNamedValue feed{"prompt", prompt};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &feed;
  inputs.num_inputs = 1;
  for (const bool half : {false, true}) {
    auto prefill = Session(true, a, SamplingModel(half)), decode = Session(true, b, SamplingModel(half));
    auto cpu_prefill = Session(false, nullptr, SamplingModel(half));
    auto cpu_decode = Session(false, nullptr, SamplingModel(half));
    for (const bool cache_gpu : {false, true}) {
      for (const bool prefill_gpu : {false, true}) {
        for (const bool decode_gpu : {false, true}) {
          for (const int64_t budget : {0, 1, 3}) {
            for (const bool eos : {false, true}) {
              auto chain = GenerationChain(prefill, decode, cache_gpu, budget, eos, prefill_gpu, decode_gpu);
              auto reference = GenerationChain(cpu_prefill, cpu_decode, false, budget, eos);
              auto execution = chain.CreateExecution(inputs), cpu = reference.CreateExecution(inputs);
              auto independent = chain.CreateExecution(inputs);
              Ort::StatefulStepResult result(nullptr);
              std::vector<std::pair<Ort::StatefulValue, std::vector<uint8_t>>> retained;
              do {
                result = execution.Step();
                auto expected = cpu.Step();
                EXPECT_EQ(result.GetInfo().executed, expected.GetInfo().executed);
                EXPECT_EQ(result.GetInfo().completed, expected.GetInfo().completed);
                ASSERT_EQ(cudaStreamQuery(a), cudaSuccess);
                ASSERT_EQ(cudaStreamQuery(b), cudaSuccess);
                if (result.GetInfo().executed) {
                  const auto logits = Named(result, "logits", ORT_STATEFUL_COMPONENT_OUTPUTS);
                  const bool gpu = result.GetInfo().component_name == "prefill" ? prefill_gpu : decode_gpu;
                  EXPECT_EQ(logits.GetTensorMemoryInfo().GetDeviceType(),
                            gpu ? OrtMemoryInfoDeviceType_GPU : OrtMemoryInfoDeviceType_CPU);
                  EXPECT_EQ(ReadBytes(logits), ReadBytes(Named(expected, "logits", ORT_STATEFUL_COMPONENT_OUTPUTS)));
                  retained.emplace_back(logits, ReadBytes(logits));
                  EXPECT_EQ(Named(result, "present_key", ORT_STATEFUL_COMPONENT_OUTPUTS)
                                .GetTensorMemoryInfo()
                                .GetDeviceType(),
                            cache_gpu ? OrtMemoryInfoDeviceType_GPU : OrtMemoryInfoDeviceType_CPU);
                }
              } while (!result.GetInfo().completed);
              const auto tokens = Named(result, "tokens");
              const size_t count = tokens.GetTensorTypeAndShapeInfo().GetElementCount();
              EXPECT_EQ(count, static_cast<size_t>(budget == 3 && eos ? 2 : budget));
              if (count) {
                EXPECT_EQ(tokens.GetTensorData<int64_t>()[0], 2);
              }
              auto state = Named(execution.GetState(), "key", ORT_STATEFUL_STATE_VALUES);
              EXPECT_EQ(Read(state), Read(Named(cpu.GetState(), "key", ORT_STATEFUL_STATE_VALUES)));
              EXPECT_TRUE(Read(Named(independent.GetState(), "key", ORT_STATEFUL_STATE_VALUES)).empty());
              for (size_t stage = 0; stage < 2; ++stage) {
                const auto info = execution.GetDispatchInfo(
                    ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, stage);
                const bool gpu = stage == 0 ? prefill_gpu : decode_gpu;
                const size_t calls = stage == 0 ? (budget ? 1 : 0) : count ? count - 1
                                                                           : 0;
                EXPECT_EQ(info.provider_calls, gpu ? calls : 0);
                EXPECT_EQ(info.cpu_calls, gpu ? 0 : calls);
                EXPECT_EQ(info.cpu_delegations, 0u);
              }
              if (budget && prefill_gpu) {
                EXPECT_EQ(execution.GetStagingInfo(0).cpu_snapshot_copy_bytes, cache_gpu ? 0u : 8u);
              }
              execution.Reset();
              execution.Run();
              for (const auto& [value, bytes] : retained) EXPECT_EQ(ReadBytes(value), bytes);
            }
          }
        }
      }
    }
  }
}

TEST_F(CudaSamplingApiIntegrationTest, GpuLogitsRemainInSelectedStateAndDownstreamFinitePorts) {
  auto model = stateful::MakeModel("logits_consumer");
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "incoming", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  stateful::SetTensorType(*graph.add_output(), "sum", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
  auto& reduce = stateful::AddNode(graph, "ReduceSum", {"incoming"}, "sum");
  auto* keepdims = reduce.add_attribute();
  keepdims->set_name("keepdims");
  keepdims->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  keepdims->set_i(0);
  auto producer = Session(true, nullptr, SamplingModel());
  auto consumer = Session(true, nullptr, model), wrong_consumer = Session(false, nullptr, model);
  auto loop = Loop(producer, false, 1, 1, false, true);
  const char *finite[]{"incoming"}, *selected[]{"sum"};
  const OrtStatefulExitCondition end{ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0};
  OrtStatefulLoopConfig consumer_config{};
  consumer_config.struct_size = sizeof(consumer_config);
  consumer_config.finite_input_names = finite;
  consumer_config.num_finite_inputs = 1;
  consumer_config.output_names = selected;
  consumer_config.num_outputs = 1;
  consumer_config.exit_conditions = &end;
  consumer_config.num_exit_conditions = 1;
  Ort::SessionLoop consumer_loop(consumer, Ort::SessionLoopOptions(consumer_config));
  auto prompt = Tokens();
  const OrtStatefulNamedValue feed{"prompt", prompt};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &feed;
  inputs.num_inputs = 1;
  const OrtStatefulInputBinding token{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
  const OrtStatefulStateBinding state{"logit_state", "logits"};
  for (const bool from_state : {false, true}) {
    const auto kind = from_state ? ORT_STATEFUL_STATE : ORT_STATEFUL_COMPONENT_OUTPUT;
    const OrtStatefulInputBinding route{"incoming", {kind, from_state ? "logit_state" : "logits", 0}};
    OrtStatefulChainComponent stages[]{
        {"prefill", nullptr, loop, &token, 1, &state, 1, nullptr, 0},
        {"consumer", from_state ? nullptr : consumer, from_state ? consumer_loop : nullptr,
         &route, 1, nullptr, 0, nullptr, 0}};
    const OrtStatefulChainOutput outputs[]{
        {"logits", 0, "logits", ORT_STATEFUL_MODEL_OUTPUT},
        {"sum", 1, "sum", ORT_STATEFUL_MODEL_OUTPUT}};
    const OrtStatefulChainConfig config{sizeof(config), stages, 2, outputs, 2};
    Ort::SessionChain chain{Ort::SessionChainOptions(config)};
    auto execution = chain.CreateExecution(inputs);
    auto result = execution.Run();
    auto logits = Named(result, "logits");
    EXPECT_EQ(logits.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
    EXPECT_EQ(Named(result, "sum").GetTensorData<float>()[0], 10.f);
    EXPECT_EQ(Named(execution.GetState(), "logit_state", ORT_STATEFUL_STATE_VALUES).GetTensorRawData(),
              logits.GetTensorRawData());
    EXPECT_EQ(execution.GetStagingInfo(1).cpu_input_request_bytes, 0u);
    EXPECT_EQ(execution.GetStagingInfo(1).cpu_snapshot_copy_bytes, 4u);
    const auto bytes = ReadBytes(logits);
    execution.Reset();
    execution.Run();
    EXPECT_EQ(ReadBytes(logits), bytes);
    stages[1].session = wrong_consumer;
    stages[1].loop = nullptr;
    EXPECT_THROW(Ort::SessionChain{Ort::SessionChainOptions(config)}, Ort::Exception);
  }
}

TEST_F(CudaSamplingApiIntegrationTest, C11GpuSamplingResultsSurviveSessionAndLibraryRelease) {
  auto session = Session();
  auto prompt = Tokens();
  OrtStatefulStepResult* raw = nullptr;
  ASSERT_EQ(StatefulApiCGpuLoopIntegrationTest(session, prompt, 1, &raw), 0);
  Ort::StatefulStepResult result(raw);
  auto logits = Named(result, "logits");
  EXPECT_EQ(logits.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  const auto bytes = ReadBytes(logits);
  result = Ort::StatefulStepResult(nullptr);
  session = Ort::Session(nullptr);
  ort_env->UnregisterExecutionProviderLibrary("CUDAExecutionProvider");
  EXPECT_EQ(ReadBytes(logits), bytes);
  logits = Ort::StatefulValue(nullptr);
  ort_env->RegisterExecutionProviderLibrary(
      "CUDAExecutionProvider", std::filesystem::path(ORT_UNIT_TEST_CUDA_PLUGIN_EP_LIBRARY_PATH).native());
}

TEST_F(CudaSamplingApiIntegrationTest, NaNsFailTransactionallyAndZeroBudgetsStillValidatePrompts) {
  auto model = SamplingModel();
  model.mutable_graph()->mutable_initializer(0)->set_float_data(0, std::numeric_limits<float>::quiet_NaN());
  auto session = Session(true, nullptr, model);
  auto loop = Loop(session, true, 1, 3, false, true);
  auto prompt = Tokens();
  const OrtStatefulNamedValue feed{"token", prompt};
  const OrtStatefulInputBatch batch{&feed, 1};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.batches = &batch;
  inputs.num_batches = 1;
  auto execution = loop.CreateExecution(inputs);
  const auto& api = Ort::GetStatefulApi();
  OrtStatefulStepResult* original = nullptr;
  Ort::ThrowOnError(api.StatefulExecutionGetState(execution, &original));
  Ort::StatefulStepResult owner(original);
  auto* unchanged = original;
  Ort::Status status(api.StatefulExecutionRun(execution, &unchanged));
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(unchanged, original);
  if (unchanged != original) api.ReleaseStatefulStepResult(unchanged);
  EXPECT_TRUE(execution.RequiresReset());
  EXPECT_THROW(execution.GetState(), Ort::Exception);
  EXPECT_EQ(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY).failures, 1u);
  EXPECT_EQ(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY).cpu_calls, 0u);
  execution.Reset();
  EXPECT_FALSE(execution.RequiresReset());
  auto healthy = Session();
  auto zero = Loop(healthy, true, 1, 0, false, true);
  for (auto invalid : {std::vector<int64_t>{}, std::vector<int64_t>{-1}}) {
    Ort::AllocatorWithDefaultOptions allocator;
    const int64_t shape[]{1, static_cast<int64_t>(invalid.size())};
    auto value = Ort::Value::CreateTensor<int64_t>(allocator, shape, 2);
    if (!invalid.empty()) {
      std::copy(invalid.begin(), invalid.end(), value.GetTensorMutableData<int64_t>());
    }
    const OrtStatefulNamedValue bad{"token", value};
    const OrtStatefulInputBatch invalid_batch{&bad, 1};
    inputs.batches = &invalid_batch;
    EXPECT_THROW(zero.CreateExecution(inputs), Ort::Exception);
  }
  inputs.batches = &batch;
  for (const int64_t position : {0, 1}) {
    auto positioned = Loop(healthy, true, 1, 1, false, true, position);
    auto result = positioned.CreateExecution(inputs).Run();
    EXPECT_EQ(result.GetInfo().generated_token, position == 0 ? 1 : 2);
    EXPECT_EQ(Named(result, "logits").GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  }
}

TEST_F(CudaSamplingApiIntegrationTest, GpuSamplingTransferAndLatency) {
  auto prefill = Session(true, nullptr, SamplingModel(false, 257));
  auto decode = Session(true, nullptr, SamplingModel(false, 257));
  auto chain = GenerationChain(prefill, decode, true, 3, false, true, true);
  auto prompt = Tokens();
  const OrtStatefulNamedValue feed{"prompt", prompt};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &feed;
  inputs.num_inputs = 1;
  auto execution = chain.CreateExecution(inputs);
  const auto start = std::chrono::steady_clock::now();
  auto result = execution.Run();
  RecordProperty("gpu_generation_run_microseconds",
                 std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
  auto tokens = Named(result, "tokens");
  ASSERT_EQ(tokens.GetTensorTypeAndShapeInfo().GetElementCount(), 3u);
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[0], 2);
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[1], 3);
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[2], 0);
  const auto prefill_info = execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 0);
  const auto decode_info = execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 1);
  const auto calls = prefill_info.provider_calls + decode_info.provider_calls;
  EXPECT_EQ(calls, 3u);
  EXPECT_EQ(execution.GetStagingInfo(0).cpu_output_bytes, 0u);
  EXPECT_EQ(execution.GetStagingInfo(1).cpu_output_bytes, 0u);
  RecordProperty("gpu_sampler_calls", calls);
}

TEST_F(CudaSamplingApiIntegrationTest, ConflictingStateFeedbackAndCrossDeviceLogitsAreRejected) {
  auto producer = Session(true, nullptr, SamplingModel());
  auto prefill = Loop(producer, false, 1, 3, false, true);
  const OrtStatefulInputBinding token{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
  const OrtStatefulStateBinding conflict{"key", "logits"};
  const OrtStatefulChainComponent writer{"prefill", nullptr, prefill, &token, 1, &conflict, 1, nullptr, 0};
  const OrtStatefulChainOutput port{"tokens", 0, nullptr, ORT_STATEFUL_GENERATED_TOKEN_IDS};
  const OrtStatefulChainConfig writer_config{sizeof(writer_config), &writer, 1, &port, 1};
  try {
    Ort::SessionChain invalid{Ort::SessionChainOptions(writer_config)};
    FAIL() << "CPU cache/GPU logits state conflict was accepted";
  } catch (const Ort::Exception& error) {
    EXPECT_NE(std::string(error.what()).find("CPU cache state writer"), std::string::npos);
  }
  auto model = stateful::MakeModel("feedback_logits_consumer");
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "incoming", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  stateful::SetTensorType(*graph.add_output(), "echo", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  stateful::AddNode(graph, "Identity", {"incoming"}, "echo");
  auto consumer = Session(true, nullptr, model);
  const char* output[]{"echo"};
  const OrtStatefulFeedbackBinding feedback{"incoming", "echo"};
  const OrtStatefulExitCondition exit{ORT_STATEFUL_MAX_ITERATIONS, 1, nullptr, nullptr, 0};
  OrtStatefulLoopConfig config{};
  config.struct_size = sizeof(config);
  config.feedback = &feedback;
  config.num_feedback = 1;
  config.output_names = output;
  config.num_outputs = 1;
  config.exit_conditions = &exit;
  config.num_exit_conditions = 1;
  Ort::SessionLoop feedback_loop(consumer, Ort::SessionLoopOptions(config));
  const OrtStatefulInputBinding route{"incoming", {ORT_STATEFUL_COMPONENT_OUTPUT, "logits", 0}};
  const OrtStatefulChainComponent stages[]{
      {"prefill", nullptr, prefill, &token, 1, nullptr, 0, nullptr, 0},
      {"feedback", nullptr, feedback_loop, &route, 1, nullptr, 0, nullptr, 0}};
  const OrtStatefulChainConfig feedback_config{sizeof(feedback_config), stages, 2, &port, 1};
  EXPECT_THROW(Ort::SessionChain{Ort::SessionChainOptions(feedback_config)}, Ort::Exception);
  int devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
  if (devices > 1) {
    auto decode_model = SamplingModel();
    auto& decode_graph = *decode_model.mutable_graph();
    stateful::SetTensorType(
        *decode_graph.add_input(), "prior_logits", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
    stateful::SetTensorType(
        *decode_graph.add_output(), "prior_seen", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
    stateful::AddNode(decode_graph, "Identity", {"prior_logits"}, "prior_seen");
    auto other = Session(true, nullptr, decode_model, 1);
    auto decode = Loop(other, false, 2);
    const OrtStatefulInputBinding previous{"prior_logits", {ORT_STATEFUL_COMPONENT_OUTPUT, "logits", 0}};
    const OrtStatefulChainComponent pipeline[]{
        {"prefill", nullptr, prefill, &token, 1, nullptr, 0, nullptr, 0},
        {"decode", nullptr, decode, &previous, 1, nullptr, 0, nullptr, 0}};
    const OrtStatefulChainConfig mixed{sizeof(mixed), pipeline, 2, &port, 1};
    Ort::SessionChainOptions options(mixed);
    options.SetComponentProvider(1, ORT_STATEFUL_FEATURE_SAMPLER, GpuCache(1));
    EXPECT_THROW(Ort::SessionChain{options}, Ort::Exception);
  }
  RecordProperty("cross_device_logits_tested", devices > 1);
}

}  // namespace
}  // namespace onnxruntime::test

#endif
