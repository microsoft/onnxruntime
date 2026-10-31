// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>

#include <cuda_runtime_api.h>
#include "nlohmann/json.hpp"
#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/stateful_execution_test_utils.h"
#include "test/util/include/ddim_test_utils.h"
#include "test/util/include/temp_dir.h"

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" int StatefulApiCGpuLoopIntegrationTest(OrtSession*, const OrtValue*, int, OrtStatefulStepResult**);
extern "C" int StatefulApiCPackageGpuChainTest(const OrtEnv*, const ORTCHAR_T*, const OrtSessionOptions*,
                                               const OrtSessionOptions*, OrtStatefulStepResult**);

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

  Ort::SessionOptions Options(bool gpu = true, cudaStream_t stream = nullptr, int id = 0, Ort::Env& env = *ort_env) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1).SetInterOpNumThreads(1).SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
    if (gpu) {
      Ort::ConstEpDevice selected(nullptr);
      for (auto device : env.GetEpDevices()) {
        if (std::strcmp(device.EpName(), "CUDAExecutionProvider") == 0 &&
            device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT).GetDeviceId() == id) selected = device;
      }
      if (!selected) ORT_CXX_API_THROW("Required CUDA plugin test device is unavailable", ORT_FAIL);
      if (stream) {
        options.AddConfigEntry("ep.cuda.user_compute_stream",
                               std::to_string(reinterpret_cast<uintptr_t>(stream)).c_str());
      }
      options.AppendExecutionProvider_V2(env, {selected}, Ort::KeyValuePairs{});
      options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    }
    return options;
  }

  Ort::Session Session(bool gpu = true, cudaStream_t stream = nullptr,
                       ONNX_NAMESPACE::ModelProto model = stateful::MakeDynamicDecoderModel(), int id = 0) {
    auto options = Options(gpu, stream, id);
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

  static ONNX_NAMESPACE::ModelProto SamplingModel(bool half = false, int64_t vocabulary = 4, bool cached = true) {
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
    if (!cached) {
      graph.mutable_input()->DeleteSubrange(1, 1);
      graph.mutable_output()->DeleteSubrange(1, 1);
      for (int i = graph.node_size() - 1; i >= 0; --i) {
        if (graph.node(i).op_type() == "Concat") graph.mutable_node()->DeleteSubrange(i, 1);
      }
    }
    return model;
  }

  Ort::SessionLoop Loop(Ort::Session& session, bool gpu = true, int phase = 0, int64_t budget = 3, bool eos = false,
                        bool gpu_sampler = false, std::optional<int64_t> position = std::nullopt, bool cached = true,
                        int sampler_device = 0) {
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
    config.num_outputs = cached ? 2 : 1;
    config.exit_conditions = phase == 2 ? &exits[1] : exits.data();
    config.num_exit_conditions = phase == 0 ? 1 : phase == 2 ? (eos ? 2 : 1)
                                                             : (eos ? 3 : 2);
    config.kv_caches = cached ? &cache : nullptr;
    config.num_kv_caches = cached ? 1 : 0;
    config.generation = phase ? &generation : nullptr;
    Ort::SessionLoopOptions options(config);
    if (gpu && cached) options.SetProvider(ORT_STATEFUL_FEATURE_CACHE, GpuCache());
    if (gpu_sampler) options.SetProvider(ORT_STATEFUL_FEATURE_SAMPLER, GpuCache(sampler_device));
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

  Ort::SessionChain BootstrapChain(Ort::Session& prefill, Ort::Session& decode, bool gpu_cache, bool gpu_sampler,
                                   int64_t budget = 3, bool eos = false, bool cached = true, bool plain = false,
                                   int sampler_device = 0) {
    auto first = Loop(prefill, gpu_cache, 0, budget, false, false, std::nullopt, cached);
    auto second = Loop(decode, gpu_cache, 2, budget, eos, gpu_sampler, std::nullopt, cached, sampler_device);
    const OrtStatefulInputBinding prompt{
        "token",
        {plain ? ORT_STATEFUL_EXTERNAL_INPUT : ORT_STATEFUL_EXTERNAL_SEQUENCE, plain ? "prompt" : "prompt_batches", 0}};
    const auto cache = CacheBinding();
    OrtStatefulChainComponent stages[]{{"prefill", nullptr, first, &prompt, 1, nullptr, 0, nullptr, 0},
                                       {"decode", nullptr, second, nullptr, 0, nullptr, 0, nullptr, 0}};
    if (plain) {
      stages[0].loop = nullptr;
      stages[0].session = prefill;
      stages[0].kv_caches = cached ? &cache : nullptr;
      stages[0].num_kv_caches = cached ? 1 : 0;
    }
    const OrtStatefulChainOutput outputs[]{{"tokens", 1, nullptr, ORT_STATEFUL_GENERATED_TOKEN_IDS},
                                           {"prefill_logits", 0, "logits", ORT_STATEFUL_MODEL_OUTPUT}};
    const OrtStatefulChainConfig config{sizeof(config), stages, 2, outputs, 2};
    Ort::SessionChainOptions options(config);
    if (plain && cached && gpu_cache) options.SetComponentProvider(0, ORT_STATEFUL_FEATURE_CACHE, GpuCache());
    const OrtStatefulGenerationBootstrapConfig bootstrap{
        sizeof(bootstrap), {ORT_STATEFUL_COMPONENT_OUTPUT, "logits", 0}, prompt.source, 0, 0};
    options.SetGenerationBootstrap(1, bootstrap);
    return Ort::SessionChain(options);
  }

  std::unique_ptr<TemporaryDirectory> Package(int64_t budget = 3, bool eos = false, bool cached = true,
                                              bool gpu_cache = true, bool gpu_sampler = true, bool half = false,
                                              int64_t vocabulary = 4, bool plain = false) {
    using Json = nlohmann::ordered_json;
    const auto root = std::filesystem::temp_directory_path() /
                      ("ort_m11_cuda_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    auto directory = std::make_unique<TemporaryDirectory>(root.native(), false);
    Json components = Json::object();
    const auto model = SamplingModel(half, vocabulary, cached);
    for (const char* stage : {"decoder_prefill", "decoder_decode"}) {
      for (const char* variant : {"cpu", "cuda"}) {
        auto variant_model = model;
        if (std::string_view(variant) == "cpu") {
          auto* scores = variant_model.mutable_graph()->mutable_initializer(0);
          for (int64_t token = 0; token < 4; ++token) {
            for (int64_t id = 0; id < vocabulary; ++id) {
              scores->set_float_data(static_cast<int>(token * vocabulary + id), id == (token + 2) % 4 ? 8.f : -1.f);
            }
          }
        }
        const bool gpu = std::string_view(variant) == "cuda";
        const std::string relative = std::string(stage) + "/" + variant;
        const char* file = std::string_view(stage) == "decoder_prefill" ? "model1.onnx" : "model2.onnx";
        std::filesystem::create_directories(root / relative);
        std::ofstream output(root / relative / file, std::ios::binary);
        ORT_ENFORCE(variant_model.SerializeToOstream(&output));
        components[stage]["variants"][variant] = {{"variant_directory", relative},
                                                  {"ep", gpu ? "CUDAExecutionProvider" : "CPUExecutionProvider"},
                                                  {"device", gpu ? "gpu" : "cpu"},
                                                  {"executor_info", {{"ort", {{"model_file", file}}}}}};
      }
    }
    const Json gpu = {{"mode", "require_provider"},
                      {"provider_name", "CUDAExecutionProvider"},
                      {"device_type", "gpu"},
                      {"device_id", 0}};
    const Json cpu = {{"mode", "cpu"}};
    Json prefill = Json::parse(R"({
      "id":"prefill","component":"decoder_prefill","executor_info":{"ort":{
        "kind":"session_loop","inputs":{"token":{"kind":"external_sequence","name":"prompt_batches"}},
        "loop":{"finite_input_names":["token"],"nonempty_input":"token",
                "exit_conditions":[{"kind":"end_of_inputs"}]}
      }}
    })");
    Json decode = Json::parse(R"({
      "id":"decode","component":"decoder_decode","executor_info":{"ort":{
        "kind":"session_loop","loop":{
          "exit_conditions":[{"kind":"max_generated_tokens","limit":3}],
          "generation":{"phase":"decode","input_ids":"token","logits_output":"logits",
            "initial_logits":{"kind":"model_output","step":"prefill","output":"logits"}},
          "sampler":{"algorithm":"greedy"}
        }
      }}
    })");
    auto& loop = decode["executor_info"]["ort"]["loop"];
    loop["exit_conditions"][0]["limit"] = budget;
    if (eos) loop["exit_conditions"].push_back({{"kind", "eos"}, {"token_ids", {3}}});
    loop["sampler"]["provider"] = gpu_sampler ? gpu : cpu;
    Json chain = {{"outputs",
                   {{"tokens", {{"kind", "generated_token_ids"}, {"step", "decode"}}},
                    {"prefill_logits", {{"kind", "model_output"}, {"step", "prefill"}, {"output", "logits"}}}}}};
    if (cached) {
      const Json bindings =
          Json::array({{{"cache", "key"}, {"input", "past_key"}, {"output", "present_key"}, {"update", "share"}}});
      prefill["executor_info"]["ort"]["loop"]["kv_caches"] = bindings;
      prefill["executor_info"]["ort"]["loop"]["cache_provider"] = gpu_cache ? gpu : cpu;
      loop["kv_caches"] = bindings;
      loop["cache_provider"] = gpu_cache ? gpu : cpu;
      chain["kv_caches"] = {
          {"key",
           {{"element_type", "float32"}, {"shape", {1, 1, 0, 1}}, {"sequence_axis", 2}, {"initial_capacity", 0}}}};
    }
    if (plain) {
      auto& ort = prefill["executor_info"]["ort"];
      ort["kind"] = "session";
      ort["inputs"]["token"] = {{"kind", "external_input"}, {"name", "prompt"}};
      if (cached) {
        ort["kv_caches"] = ort["loop"]["kv_caches"];
        ort["cache_provider"] = ort["loop"]["cache_provider"];
      }
      ort.erase("loop");
      loop["generation"]["initial_logits"]["prompt_input"] = "token";
    }
    const Json manifest = {
        {"schema_version", "1.1"},
        {"layout", "portable"},
        {"components", components},
        {"chains", {{"generate", {{"steps", Json::array({prefill, decode})}, {"executor_info", {{"ort", chain}}}}}}}};
    std::ofstream(root / "manifest.json") << manifest.dump(2);
    return directory;
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

class CudaBootstrapPackageTest : public CudaCacheApiIntegrationTest {};

TEST_F(CudaBootstrapPackageTest, PackageProgrammaticCpuParityAndIndependentPolicies) {
  cudaStream_t a = nullptr, b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
  auto release = [](cudaStream_t stream) { EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess); };
  using StreamOwner = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(release)>;
  StreamOwner first_stream(a, release), second_stream(b, release);
  auto first_options = Options(true, a), second_options = Options(true, b);
  auto prompt = Tokens();
  const OrtValue* batches[]{prompt};
  const OrtStatefulInputSequence sequence{"prompt_batches", batches, 1};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.sequences = &sequence;
  inputs.num_sequences = 1;
  size_t cases = 0;
  for (bool cached : {false, true}) {
    for (bool half : {false, true}) {
      auto prefill = Session(true, a, SamplingModel(half, 4, cached));
      auto decode = Session(true, b, SamplingModel(half, 4, cached));
      auto cpu_prefill = Session(false, nullptr, SamplingModel(half, 4, cached));
      auto cpu_decode = Session(false, nullptr, SamplingModel(half, 4, cached));
      for (bool gpu_cache : {false, true}) {
        if (!cached && gpu_cache) continue;
        for (bool gpu_sampler : {false, true}) {
          for (int64_t budget : {0, 1, 3}) {
            for (bool eos : {false, true}) {
              auto files = Package(budget, eos, cached, gpu_cache, gpu_sampler, half);
              Ort::ModelPackageContext package(files->Path().c_str());
              const OrtModelPackageStepSessionOptions steps[]{{"prefill", first_options}, {"decode", second_options}};
              const OrtModelPackageChainConfig config{sizeof(config), "generate", nullptr, steps, 2};
              auto loaded = package.CreateSessionChain(*ort_env, config);
              auto direct = BootstrapChain(prefill, decode, gpu_cache, gpu_sampler, budget, eos, cached);
              auto reference = BootstrapChain(cpu_prefill, cpu_decode, false, false, budget, eos, cached);
              auto execution = loaded.CreateExecution(inputs);
              auto programmatic = direct.CreateExecution(inputs);
              auto cpu = reference.CreateExecution(inputs);
              auto independent = loaded.CreateExecution(inputs);
              package = Ort::ModelPackageContext(nullptr);
              loaded = Ort::SessionChain(nullptr);
              Ort::StatefulStepResult result(nullptr);
              Ort::StatefulStepResult expected(nullptr);
              std::vector<std::pair<Ort::StatefulValue, std::vector<uint8_t>>> retained;
              do {
                result = execution.Step();
                auto native = programmatic.Step();
                expected = cpu.Step();
                const auto info = result.GetInfo(), other = expected.GetInfo(), native_info = native.GetInfo();
                EXPECT_EQ(info.executed, other.executed);
                EXPECT_EQ(info.completed, other.completed);
                EXPECT_EQ(info.invocations, other.invocations);
                EXPECT_EQ(info.iterations, other.iterations);
                EXPECT_EQ(info.generated_token, other.generated_token);
                EXPECT_EQ(info.exit_reasons, other.exit_reasons);
                EXPECT_EQ(info.invocations, native_info.invocations);
                EXPECT_EQ(info.generated_token, native_info.generated_token);
                ASSERT_EQ(cudaStreamQuery(a), cudaSuccess);
                ASSERT_EQ(cudaStreamQuery(b), cudaSuccess);
                if (info.executed) {
                  auto logits = Named(result, "logits", ORT_STATEFUL_COMPONENT_OUTPUTS);
                  EXPECT_EQ(logits.GetTensorMemoryInfo().GetDeviceType(),
                            gpu_sampler ? OrtMemoryInfoDeviceType_GPU : OrtMemoryInfoDeviceType_CPU);
                  EXPECT_EQ(ReadBytes(logits), ReadBytes(Named(expected, "logits", ORT_STATEFUL_COMPONENT_OUTPUTS)));
                  EXPECT_EQ(ReadBytes(logits), ReadBytes(Named(native, "logits", ORT_STATEFUL_COMPONENT_OUTPUTS)));
                  retained.emplace_back(logits, ReadBytes(logits));
                  if (cached) {
                    EXPECT_EQ(Named(result, "present_key", ORT_STATEFUL_COMPONENT_OUTPUTS)
                                  .GetTensorMemoryInfo()
                                  .GetDeviceType(),
                              gpu_cache ? OrtMemoryInfoDeviceType_GPU : OrtMemoryInfoDeviceType_CPU);
                  }
                } else if (info.generated_token) {
                  EXPECT_EQ(info.component_name, "decode");
                  EXPECT_EQ(info.iterations, 0);
                  EXPECT_EQ(info.invocations, 1);
                  EXPECT_TRUE(result.GetOutputs(ORT_STATEFUL_COMPONENT_OUTPUTS).empty());
                }
              } while (!result.GetInfo().completed);
              const auto count = static_cast<size_t>(eos && budget == 3 ? 2 : budget);
              auto tokens = Named(result, "tokens");
              EXPECT_EQ(tokens.GetTensorTypeAndShapeInfo().GetElementCount(), count);
              EXPECT_EQ(ReadBytes(tokens), ReadBytes(Named(expected, "tokens")));
              const auto sampler =
                  execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 1);
              EXPECT_EQ(sampler.provider_calls, gpu_sampler ? count : 0);
              EXPECT_EQ(sampler.cpu_calls, gpu_sampler ? 0 : count);
              EXPECT_EQ(sampler.cpu_delegations, 0u);
              EXPECT_THROW(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 0),
                           Ort::Exception);
              if (cached) {
                auto state = Named(execution.GetState(), "key", ORT_STATEFUL_STATE_VALUES);
                EXPECT_EQ(Read(state), Read(Named(cpu.GetState(), "key", ORT_STATEFUL_STATE_VALUES)));
                EXPECT_EQ(state.GetTensorTypeAndShapeInfo().GetShape()[2],
                          static_cast<int64_t>(2 + (count ? count - 1 : 0)));
                EXPECT_TRUE(Read(Named(independent.GetState(), "key", ORT_STATEFUL_STATE_VALUES)).empty());
                const auto cache = execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_CACHE, ORT_STATEFUL_CACHE_SHARE);
                EXPECT_EQ(cache.provider_calls, gpu_cache ? std::max<size_t>(1, count) : 0);
                EXPECT_EQ(cache.cpu_calls, gpu_cache ? 0 : std::max<size_t>(1, count));
              }
              if (gpu_sampler && (gpu_cache || !cached)) {
                EXPECT_EQ(execution.GetStagingInfo(0).cpu_output_bytes, 0u);
                EXPECT_EQ(execution.GetStagingInfo(1).cpu_output_bytes, 0u);
              }
              execution.Reset();
              EXPECT_EQ(ReadBytes(Named(execution.Run(), "tokens")), ReadBytes(tokens));
              for (const auto& [value, bytes] : retained) EXPECT_EQ(ReadBytes(value), bytes);
              ++cases;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(cases, 72u);
  RecordProperty("bootstrap_policy_dtype_budget_eos_cases", cases);
}

TEST_F(CudaBootstrapPackageTest, PlainPrefillFirstEosCancellationAndInvalidProducers) {
  auto options = Options();
  auto files = Package(3, true, true, true, true, false, 4, true);
  Ort::ModelPackageContext package(files->Path().c_str());
  const OrtModelPackageChainConfig config{sizeof(config), "generate", options, nullptr, 0};
  auto chain = package.CreateSessionChain(*ort_env, config);
  auto prompt = Tokens({0, 2});
  const OrtStatefulNamedValue feed{"prompt", prompt};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &feed;
  inputs.num_inputs = 1;
  auto execution = chain.CreateExecution(inputs);
  auto first = execution.Step();
  EXPECT_TRUE(first.GetInfo().executed);
  execution.Cancel();
  EXPECT_THROW(execution.Step(), Ort::Exception);
  EXPECT_TRUE(execution.RequiresReset());
  EXPECT_EQ(execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 1).provider_calls, 0u);
  execution.Reset();
  auto result = execution.Run();
  EXPECT_TRUE(result.GetInfo().completed);
  EXPECT_FALSE(result.GetInfo().executed);
  EXPECT_EQ(result.GetInfo().generated_token, 3);
  EXPECT_EQ(result.GetInfo().invocations, 1);
  EXPECT_EQ(result.GetInfo().generated_tokens, 1);
  auto retained = Named(result, "prefill_logits");
  const auto bytes = ReadBytes(retained);
  execution.Reset();
  EXPECT_EQ(ReadBytes(retained), bytes);

  auto host = Session(false, nullptr, SamplingModel(false, 4, false));
  auto gpu = Session(true, nullptr, SamplingModel(false, 4, false));
  EXPECT_THROW(BootstrapChain(host, gpu, false, true, 3, false, false), Ort::Exception);
  int devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
  if (devices > 1) {
    auto other = Session(true, nullptr, SamplingModel(false, 4, false), 1);
    EXPECT_THROW(BootstrapChain(gpu, other, false, true, 3, false, false, false, 1), Ort::Exception);
  }
  auto nan = SamplingModel();
  nan.mutable_graph()->mutable_initializer(0)->set_float_data(0, std::numeric_limits<float>::quiet_NaN());
  auto bad = Session(true, nullptr, nan);
  auto decode = Session();
  auto failing = BootstrapChain(bad, decode, true, true);
  auto token = Tokens();
  const OrtValue* batches[]{token};
  const OrtStatefulInputSequence sequence{"prompt_batches", batches, 1};
  OrtStatefulInputs finite{};
  finite.struct_size = sizeof(finite);
  finite.sequences = &sequence;
  finite.num_sequences = 1;
  auto failed = failing.CreateExecution(finite);
  failed.Step();
  EXPECT_THROW(failed.Step(), Ort::Exception);
  EXPECT_TRUE(failed.RequiresReset());
  const auto dispatch = failed.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 1);
  EXPECT_EQ(dispatch.provider_calls, 1u);
  EXPECT_EQ(dispatch.failures, 1u);
  EXPECT_EQ(dispatch.cpu_calls, 0u);
  EXPECT_EQ(dispatch.cpu_delegations, 0u);
  failed.Reset();
  RecordProperty("cross_device_bootstrap_tested", devices > 1);
}

TEST_F(CudaBootstrapPackageTest, C11AndGpuValuesSurviveOwnedSessionsAndLibraryUnregistration) {
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "m11-owned-cuda");
  env.RegisterExecutionProviderLibrary("m11-cuda", ORT_UNIT_TEST_CUDA_PLUGIN_EP_LIBRARY_PATH);
  auto options = Options(true, nullptr, 0, env);
  auto files = Package();
  OrtStatefulStepResult* raw = nullptr;
  ASSERT_EQ(StatefulApiCPackageGpuChainTest(env, files->Path().c_str(), options, options, &raw), 0);
  Ort::StatefulStepResult result(raw);
  auto logits = Named(result, "prefill_logits");
  EXPECT_EQ(logits.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  const auto bytes = ReadBytes(logits);
  options = Ort::SessionOptions(nullptr);
  result = Ort::StatefulStepResult(nullptr);
  env.UnregisterExecutionProviderLibrary("m11-cuda");
  EXPECT_EQ(ReadBytes(logits), bytes);
  logits = Ort::StatefulValue(nullptr);
}

class CudaDdimPackageTest : public CudaCacheApiIntegrationTest {
 protected:
  class Inputs {
   public:
    Inputs(const DdimOptions& policy, bool seeded = false) {
      Ort::AllocatorWithDefaultOptions allocator;
      initial = seeded ? Ort::Value::CreateTensor<uint64_t>(allocator, nullptr, 0)
                       : Ort::Value::CreateTensor<float>(allocator, policy.latent_shape.data(), 4);
      if (seeded) {
        *initial.GetTensorMutableData<uint64_t>() = 0;
      } else {
        const auto count = initial.GetTensorTypeAndShapeInfo().GetElementCount();
        const std::array<float, 3> samples{0.125f, -0.25f, 1.f};
        for (size_t i = 0; i < count; ++i) initial.GetTensorMutableData<float>()[i] = samples[i % samples.size()];
      }
      const std::array<int64_t, 4> shape{policy.latent_shape[0], 1, 1, 1};
      positive = Ort::Value::CreateTensor<float>(allocator, shape.data(), 4);
      negative = Ort::Value::CreateTensor<float>(allocator, shape.data(), 4);
      std::fill_n(positive.GetTensorMutableData<float>(), shape[0], 0.25f);
      std::fill_n(negative.GetTensorMutableData<float>(), shape[0], -0.5f);
    }

    OrtStatefulInputs Description() {
      named = {{{"init", initial}, {"positive", positive}, {"negative", negative}}};
      OrtStatefulInputs inputs{};
      inputs.struct_size = sizeof(inputs);
      inputs.inputs = named.data();
      inputs.num_inputs = named.size();
      return inputs;
    }
    Ort::Value initial{nullptr}, positive{nullptr}, negative{nullptr};
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Inputs);

   private:
    std::array<OrtStatefulNamedValue, 3> named;
  };

  std::unique_ptr<TemporaryDirectory> DdimPackage(const DdimOptions& policy) {
    using Json = nlohmann::ordered_json;
    const auto root = std::filesystem::temp_directory_path() /
                      ("ort_m12_ddim_cuda_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    auto directory = std::make_unique<TemporaryDirectory>(root.native(), false);
    Json manifest{{"schema_version", "1.1"}};
    for (const char* variant : {"cpu", "cuda"}) {
      const bool gpu = std::string_view(variant) == "cuda";
      auto model = stateful::MakeDdimDenoiserModel(policy);
      if (!gpu) model.mutable_graph()->mutable_initializer(0)->set_float_data(0, 0.25f);
      const std::string relative = std::string("denoiser/") + variant;
      std::filesystem::create_directories(root / relative);
      std::ofstream output(root / relative / "denoiser.onnx", std::ios::binary);
      ORT_ENFORCE(model.SerializeToOstream(&output));
      manifest["components"]["denoiser"]["variants"][variant] = {
          {"variant_directory", relative},
          {"ep", gpu ? "CUDAExecutionProvider" : "CPUExecutionProvider"},
          {"device", gpu ? "gpu" : "cpu"},
          {"executor_info", {{"ort", {{"model_file", "denoiser.onnx"}}}}}};
    }
    manifest["chains"]["denoise"] = stateful::DdimChainJson(policy);
    std::ofstream(root / "manifest.json") << manifest.dump(2);
    return directory;
  }

  Ort::SessionLoop DdimLoop(Ort::Session& session, const DdimOptions& policy) {
    const char* names[]{"latents"};
    const OrtStatefulExitCondition exit{ORT_STATEFUL_MAX_ITERATIONS, policy.num_inference_steps, nullptr, nullptr, 0};
    OrtStatefulLoopConfig config{};
    config.struct_size = sizeof(config);
    config.output_names = names;
    config.num_outputs = 1;
    config.exit_conditions = &exit;
    config.num_exit_conditions = 1;
    Ort::SessionLoopOptions options(config);
    options.SetDdim(stateful::PublicDdimConfig(policy));
    return Ort::SessionLoop(session, options);
  }

  void Transfer(DdimGuidanceMode mode) {
    auto policy = stateful::MakeDdimOptions(3);
    policy.latent_shape = {2, 2, 3, 5};
    policy.guidance_mode = mode;
    policy.timestep_format = DdimTimestepFormat::Float32Vector;
    auto files = DdimPackage(policy);
    auto options = Options();
    Ort::ModelPackageContext package(files->Path().c_str());
    const OrtModelPackageChainConfig config{sizeof(config), "denoise", options, nullptr, 0};
    auto chain = package.CreateSessionChain(*ort_env, config);
    Inputs values(policy);
    const auto inputs = values.Description();
    auto execution = chain.CreateExecution(inputs);
    auto result = execution.Run();
    const auto latent = Named(result, "latents");
    EXPECT_EQ(latent.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
    EXPECT_EQ(latent.GetTensorTypeAndShapeInfo().GetElementCount(), 60u);
    EXPECT_EQ(result.GetInfo().invocations, mode == DdimGuidanceMode::Batched ? 3 : 6);
    const auto staging = execution.GetStagingInfo(0);
    EXPECT_EQ(staging.cpu_output_bytes, 1440u);
    EXPECT_EQ(staging.cpu_snapshot_copy_bytes, 1440u);
    EXPECT_TRUE(std::isfinite(latent.GetTensorData<float>()[0]));
  }
};

TEST_F(CudaDdimPackageTest, CpuProgrammaticPackageParityGuidanceSeedsAndNondefaultStream) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  auto release = [](cudaStream_t value) { EXPECT_EQ(cudaStreamDestroy(value), cudaSuccess); };
  std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(release)> stream_owner(stream, release);
  for (const auto mode : {DdimGuidanceMode::Batched, DdimGuidanceMode::Sequential}) {
    for (const float scale : {0.f, 1.f, 7.5f}) {
      auto policy = stateful::MakeDdimOptions(3);
      policy.guidance_mode = mode;
      policy.guidance_scale = scale;
      policy.latent_shape = {2, 2, 3, 5};
      policy.timestep_format = DdimTimestepFormat::Float32Vector;
      auto gpu = Session(true, stream, stateful::MakeDdimDenoiserModel(policy));
      auto cpu = Session(false, nullptr, stateful::MakeDdimDenoiserModel(policy));
      auto gpu_loop = DdimLoop(gpu, policy);
      auto cpu_loop = DdimLoop(cpu, policy);
      auto files = DdimPackage(policy);
      auto options = Options(true, stream);
      Ort::ModelPackageContext package(files->Path().c_str());
      const OrtModelPackageChainConfig config{sizeof(config), "denoise", options, nullptr, 0};
      auto chain = package.CreateSessionChain(*ort_env, config);
      for (const bool seeded : {false, true}) {
        Inputs values(policy, seeded);
        const auto inputs = values.Description();
        auto execution = chain.CreateExecution(inputs);
        auto result = execution.Run();
        auto expected = cpu_loop.CreateExecution(inputs).Run();
        auto programmatic = gpu_loop.CreateExecution(inputs).Run();
        auto latent = Named(result, "latents");
        const auto reference = Named(expected, "latents");
        const auto gpu_value = Named(programmatic, "latents");
        ASSERT_EQ(latent.GetTensorTypeAndShapeInfo().GetElementCount(), 60u);
        EXPECT_EQ(latent.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
        EXPECT_EQ(result.GetInfo().invocations, mode == DdimGuidanceMode::Batched ? 3 : 6);
        for (size_t i = 0; i < 60; ++i) {
          const float target = reference.GetTensorData<float>()[i];
          const float tolerance = 1e-4f + 1e-4f * std::abs(target);
          EXPECT_NEAR(latent.GetTensorData<float>()[i], target, tolerance);
          EXPECT_NEAR(gpu_value.GetTensorData<float>()[i], target, tolerance);
        }
        execution.Reset();
        EXPECT_EQ(Named(execution.Run(), "latents").GetTensorData<float>()[0], latent.GetTensorData<float>()[0]);
      }
    }
  }
}

TEST_F(CudaDdimPackageTest, CpuLatentsSurviveOwnedSessionsAndLibraryUnregistration) {
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "m12-owned-cuda");
  env.RegisterExecutionProviderLibrary("m12-cuda", ORT_UNIT_TEST_CUDA_PLUGIN_EP_LIBRARY_PATH);
  auto options = Options(true, nullptr, 0, env);
  auto policy = stateful::MakeDdimOptions(2);
  auto files = DdimPackage(policy);
  Ort::ModelPackageContext package(files->Path().c_str());
  const OrtModelPackageChainConfig config{sizeof(config), "denoise", options, nullptr, 0};
  auto chain = package.CreateSessionChain(env, config);
  Inputs values(policy);
  const auto inputs = values.Description();
  auto execution = chain.CreateExecution(inputs);
  auto result = execution.Run();
  auto latent = Named(result, "latents");
  execution.Reset();
  execution = Ort::StatefulExecution(nullptr);
  chain = Ort::SessionChain(nullptr);
  package = Ort::ModelPackageContext(nullptr);
  result = Ort::StatefulStepResult(nullptr);
  options = Ort::SessionOptions(nullptr);
  env.UnregisterExecutionProviderLibrary("m12-cuda");
  EXPECT_EQ(latent.GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
  EXPECT_NEAR(latent.GetTensorData<float>()[0], -8.022068f, 1e-4f + 1e-4f * 8.022068f);
}

TEST_F(CudaDdimPackageTest, DeviceLogicalInitializationIsRejectedWithoutHiddenReadback) {
  auto policy = stateful::MakeDdimOptions(2);
  auto session = Session(true, nullptr, stateful::MakeDdimDenoiserModel(policy));
  auto loop = DdimLoop(session, policy);
  Inputs values(policy);
  void* pointer = nullptr;
  ASSERT_EQ(cudaMalloc(&pointer, 3 * sizeof(float)), cudaSuccess);
  auto release = [](void* value) { EXPECT_EQ(cudaFree(value), cudaSuccess); };
  std::unique_ptr<void, decltype(release)> owner(pointer, release);
  const Ort::MemoryInfo memory("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  values.initial = Ort::Value::CreateTensor<float>(memory, static_cast<float*>(pointer), 3,
                                                   policy.latent_shape.data(), 4);
  const auto inputs = values.Description();
  EXPECT_THROW(loop.CreateExecution(inputs), Ort::Exception);
}

TEST_F(CudaDdimPackageTest, TransferBatched) { Transfer(DdimGuidanceMode::Batched); }
TEST_F(CudaDdimPackageTest, TransferSequential) { Transfer(DdimGuidanceMode::Sequential); }

TEST_F(CudaBootstrapPackageTest, BootstrapPackageTransfer) {
  auto files = Package(3, false, true, true, true, false, 257);
  auto options = Options();
  Ort::ModelPackageContext package(files->Path().c_str());
  const OrtModelPackageChainConfig config{sizeof(config), "generate", options, nullptr, 0};
  auto chain = package.CreateSessionChain(*ort_env, config);
  auto prompt = Tokens();
  const OrtValue* batches[]{prompt};
  const OrtStatefulInputSequence sequence{"prompt_batches", batches, 1};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.sequences = &sequence;
  inputs.num_sequences = 1;
  auto execution = chain.CreateExecution(inputs);
  auto result = execution.Run();
  const auto tokens = Named(result, "tokens");
  ASSERT_EQ(tokens.GetTensorTypeAndShapeInfo().GetElementCount(), 3u);
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[0], 2);
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[1], 3);
  EXPECT_EQ(tokens.GetTensorData<int64_t>()[2], 0);
  EXPECT_EQ(Named(result, "prefill_logits").GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  const auto sampler = execution.GetDispatchInfo(ORT_STATEFUL_FEATURE_SAMPLER, ORT_STATEFUL_SAMPLE_GREEDY, 1);
  EXPECT_EQ(sampler.provider_calls, 3u);
  EXPECT_EQ(sampler.cpu_calls, 0u);
  EXPECT_EQ(sampler.cpu_delegations, 0u);
  EXPECT_EQ(execution.GetStagingInfo(0).cpu_output_bytes, 0u);
  EXPECT_EQ(execution.GetStagingInfo(1).cpu_output_bytes, 0u);
  RecordProperty("gpu_bootstrap_sampler_calls", sampler.provider_calls);
}

}  // namespace
}  // namespace onnxruntime::test

#endif
