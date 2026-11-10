// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <string_view>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/stateful_execution_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;
extern "C" int StatefulApiCConsumerTest(void);

namespace onnxruntime::test {
namespace {

const OrtStatefulApi& Api() {
  return *Ort::GetApi().GetStatefulApi();
}

void Ok(OrtStatus* status) {
  Ort::ThrowOnError(status);
}

void Error(OrtStatus* status) {
  ASSERT_NE(status, nullptr);
  EXPECT_NE(Ort::GetApi().GetErrorCode(status), ORT_OK);
  EXPECT_NE(std::string_view(Ort::GetApi().GetErrorMessage(status)).size(), 0u);
  Ort::GetApi().ReleaseStatus(status);
}

template <typename T>
using Handle = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T, typename Release>
Handle<T> Own(T* value, Release release) {
  return Handle<T>(value, release);
}

template <typename T>
Ort::Value Tensor(std::initializer_list<int64_t> shape, std::initializer_list<T> values) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto result = Ort::Value::CreateTensor<T>(allocator, shape.begin(), shape.size());
  if (values.size() != 0) {
    std::copy(values.begin(), values.end(), result.template GetTensorMutableData<T>());
  }
  return result;
}

const OrtValue* Output(const OrtStatefulStepResult* result, const char* name,
                       OrtStatefulOutputGroup group = ORT_STATEFUL_SELECTED_OUTPUTS) {
  const OrtStatefulNamedValue* values = nullptr;
  size_t count = 0;
  Ok(Api().StatefulStepResultGetOutputs(result, group, &values, &count));
  for (size_t i = 0; i < count; ++i) {
    if (std::string_view(values[i].name) == name) {
      return values[i].value;
    }
  }
  ADD_FAILURE() << "Output not found: " << name;
  return nullptr;
}

float Float(const OrtValue* value) {
  const void* data = nullptr;
  Ok(Ort::GetApi().GetTensorData(value, &data));
  return *static_cast<const float*>(data);
}

OrtStatefulStepInfo Info(const OrtStatefulStepResult* result) {
  OrtStatefulStepInfo info{};
  Ok(Api().StatefulStepResultGetInfo(result, sizeof(info), &info));
  return info;
}

struct FeedbackConfig {
  const char* finite[1]{"input_value"};
  const char* outputs[2]{"present_value", "done"};
  OrtStatefulFeedbackBinding feedback[1]{{"past_value", "present_value"}};
  OrtStatefulExitCondition exits[3]{
      {ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0},
      {ORT_STATEFUL_MAX_ITERATIONS, 3, nullptr, nullptr, 0},
      {ORT_STATEFUL_BOOLEAN_OUTPUT, 0, "done", nullptr, 0},
  };
  OrtStatefulLoopConfig config{};
  FeedbackConfig() {
    config.struct_size = sizeof(config);
    config.finite_input_names = finite;
    config.num_finite_inputs = 1;
    config.feedback = feedback;
    config.num_feedback = 1;
    config.output_names = outputs;
    config.num_outputs = 2;
    config.exit_conditions = exits;
    config.num_exit_conditions = 3;
    config.nonempty_input = "input_value";
  }
};

class StatefulApiTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NE(Ort::GetApi().GetStatefulApi(), nullptr);
    EXPECT_EQ(Ort::GetApi().GetStatefulApi(), Ort::GetApi().GetStatefulApi());
  }
  Ort::Session Session(const ONNX_NAMESPACE::ModelProto& model, bool cuda = false) {
    const auto bytes = model.SerializeAsString();
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
    if (cuda) {
      options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
      options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    }
    Ort::Session session(*ort_env, bytes.data(), bytes.size(), options);
    const auto subgraphs = session.GetEpGraphAssignmentInfo();
    EXPECT_FALSE(subgraphs.empty());
    for (const auto& subgraph : subgraphs) {
      EXPECT_EQ(subgraph.GetEpName(), cuda ? "CUDAExecutionProvider" : "CPUExecutionProvider");
    }
    return session;
  }
  Handle<OrtSessionLoopOptions> LoopOptions(const OrtStatefulLoopConfig& config) {
    OrtSessionLoopOptions* value = nullptr;
    Ok(Api().CreateSessionLoopOptions(&config, &value));
    return Own(value, Api().ReleaseSessionLoopOptions);
  }
  Handle<OrtSessionLoop> Loop(OrtSession* session, const OrtStatefulLoopConfig& config) {
    auto options = LoopOptions(config);
    OrtSessionLoop* value = nullptr;
    Ok(Api().CreateSessionLoop(session, options.get(), &value));
    return Own(value, Api().ReleaseSessionLoop);
  }
};

class StatefulApiProviderTest : public StatefulApiTest, public testing::WithParamInterface<bool> {};

TEST(StatefulApiAbiTest, AccessorIsAppendedAndReleaseAcceptsNull) {
  EXPECT_EQ(offsetof(OrtApi, GetStatefulApi), offsetof(OrtApi, GetModelPackageApi) + sizeof(void*));
  const auto& api = Api();
  api.ReleaseSessionLoopOptions(nullptr);
  api.ReleaseSessionLoop(nullptr);
  api.ReleaseSessionChainOptions(nullptr);
  api.ReleaseSessionChain(nullptr);
  api.ReleaseStatefulExecution(nullptr);
  api.ReleaseStatefulStepResult(nullptr);
  api.ReleaseKVCacheOptions(nullptr);
  api.ReleaseKVCacheManager(nullptr);
  api.ReleaseKVCacheView(nullptr);
  api.ReleaseSamplerOptions(nullptr);
}

TEST(StatefulApiAbiTest, PureCConsumer) {
  EXPECT_EQ(StatefulApiCConsumerTest(), 0);
}

TEST_P(StatefulApiProviderTest, LoopStateSteppingResetAndRetainedResults) {
  auto session = Session(stateful::MakeFeedbackModel(), GetParam());
  FeedbackConfig description;
  std::string finite = "input_value", selected = "present_value";
  description.finite[0] = finite.c_str();
  description.outputs[0] = selected.c_str();
  auto options = LoopOptions(description.config);
  finite = "changed";
  selected = "changed";
  OrtSessionLoop* raw_loop = nullptr;
  Ok(Api().CreateSessionLoop(session, options.get(), &raw_loop));
  auto loop = Own(raw_loop, Api().ReleaseSessionLoop);
  FeedbackConfig replacement;
  replacement.config.num_exit_conditions = 1;
  replacement.exits[0] = {ORT_STATEFUL_MAX_ITERATIONS, 0, nullptr, nullptr, 0};
  Ok(Api().SessionLoopOptionsSetConfig(options.get(), &replacement.config));
  auto invalid = replacement.config;
  invalid.struct_size = 0;
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  OrtSessionLoop* empty_loop = nullptr;
  Ok(Api().CreateSessionLoop(session, options.get(), &empty_loop));
  auto zero = Own(empty_loop, Api().ReleaseSessionLoop);
  options.reset();

  auto initial = Tensor<float>({}, {0.f});
  std::array<Ort::Value, 3> values{Tensor<float>({}, {2.f}), Tensor<float>({}, {3.f}), Tensor<float>({}, {4.f})};
  OrtStatefulNamedValue invariant{"past_value", initial};
  std::array<OrtStatefulNamedValue, 3> finite_values{
      OrtStatefulNamedValue{"input_value", values[0]},
      OrtStatefulNamedValue{"input_value", values[1]},
      OrtStatefulNamedValue{"input_value", values[2]},
  };
  std::array<OrtStatefulInputBatch, 3> batches{
      OrtStatefulInputBatch{&finite_values[0], 1},
      OrtStatefulInputBatch{&finite_values[1], 1},
      OrtStatefulInputBatch{&finite_values[2], 1},
  };
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &invariant;
  inputs.num_inputs = 1;
  inputs.batches = batches.data();
  inputs.num_batches = batches.size();
  OrtStatefulExecution* raw_execution = nullptr;
  Ok(Api().CreateLoopExecution(loop.get(), &inputs, &raw_execution));
  auto execution = Own(raw_execution, Api().ReleaseStatefulExecution);
  Ok(Api().CreateLoopExecution(zero.get(), &inputs, &raw_execution));
  auto zero_execution = Own(raw_execution, Api().ReleaseStatefulExecution);
  initial = Ort::Value(nullptr);
  for (auto& value : values) {
    value = Ort::Value(nullptr);
  }
  loop.reset();
  zero.reset();
  OrtStatefulStepResult* raw_result = nullptr;
  Ok(Api().StatefulExecutionGetState(execution.get(), &raw_result));
  auto original_state = Own(raw_result, Api().ReleaseStatefulStepResult);
  EXPECT_EQ(Info(original_state.get()).kind, ORT_STATEFUL_STATE_SNAPSHOT);
  EXPECT_EQ(Float(Output(original_state.get(), "past_value", ORT_STATEFUL_STATE_VALUES)), 0.f);
  Handle<OrtStatefulStepResult> first(nullptr, Api().ReleaseStatefulStepResult);
  for (int64_t i = 0; i < 3; ++i) {
    Ok(Api().StatefulExecutionStep(execution.get(), &raw_result));
    auto result = Own(raw_result, Api().ReleaseStatefulStepResult);
    auto info = Info(result.get());
    EXPECT_EQ(info.kind, ORT_STATEFUL_LOOP_RESULT);
    EXPECT_EQ(info.iterations, i + 1);
    EXPECT_EQ(info.invocations, i + 1);
    EXPECT_TRUE(info.executed);
    EXPECT_EQ(Float(Output(result.get(), "present_value")), i == 0 ? 2.f : i == 1 ? 5.f
                                                                                  : 9.f);
    EXPECT_EQ(Float(Output(result.get(), "present_value", ORT_STATEFUL_COMPONENT_OUTPUTS)),
              Float(Output(result.get(), "present_value")));
    if (i == 0) {
      first = std::move(result);
    }
    if (i == 2) {
      EXPECT_TRUE(info.completed);
      ASSERT_EQ(info.num_exit_reasons, 3u);
      EXPECT_EQ(info.exit_reasons[0], ORT_STATEFUL_END_OF_INPUTS);
      EXPECT_EQ(info.exit_reasons[1], ORT_STATEFUL_MAX_ITERATIONS);
      EXPECT_EQ(info.exit_reasons[2], ORT_STATEFUL_BOOLEAN_OUTPUT);
      struct ExtendedInfo {
        OrtStatefulStepInfo value;
        uint64_t tail;
      } extended{};
      extended.tail = 0x12345678;
      Ok(Api().StatefulStepResultGetInfo(result.get(), sizeof(extended), &extended.value));
      EXPECT_EQ(extended.value.struct_size, sizeof(OrtStatefulStepInfo));
      EXPECT_EQ(extended.tail, 0x12345678u);
    }
  }
  Ok(Api().StatefulExecutionGetState(execution.get(), &raw_result));
  auto current_state = Own(raw_result, Api().ReleaseStatefulStepResult);
  EXPECT_EQ(Float(Output(current_state.get(), "past_value", ORT_STATEFUL_STATE_VALUES)), 9.f);
  EXPECT_EQ(Float(Output(original_state.get(), "past_value", ORT_STATEFUL_STATE_VALUES)), 0.f);
  Ok(Api().StatefulExecutionStep(execution.get(), &raw_result));
  auto done = Own(raw_result, Api().ReleaseStatefulStepResult);
  EXPECT_FALSE(Info(done.get()).executed);
  const OrtStatefulNamedValue* outputs = nullptr;
  size_t count = 77;
  Ok(Api().StatefulStepResultGetOutputs(done.get(), ORT_STATEFUL_SELECTED_OUTPUTS, &outputs, &count));
  EXPECT_EQ(outputs, nullptr);
  EXPECT_EQ(count, 0u);
  Ok(Api().StatefulExecutionReset(execution.get()));
  Ok(Api().StatefulExecutionRun(execution.get(), &raw_result));
  auto rerun = Own(raw_result, Api().ReleaseStatefulStepResult);
  EXPECT_EQ(Float(Output(rerun.get(), "present_value")), 9.f);
  Ok(Api().StatefulExecutionRun(zero_execution.get(), &raw_result));
  auto skipped = Own(raw_result, Api().ReleaseStatefulStepResult);
  EXPECT_FALSE(Info(skipped.get()).executed);
  EXPECT_EQ(Info(skipped.get()).iterations, 0);
  execution.reset();
  zero_execution.reset();
  session = Ort::Session(nullptr);
  EXPECT_EQ(Float(Output(first.get(), "present_value")), 2.f);
}

TEST_P(StatefulApiProviderTest, ChainsExposeBindingsSequencesStateAndBothComponentKinds) {
  for (bool first_loop : {false, true}) {
    for (bool second_loop : {false, true}) {
      auto first_session = Session(stateful::MakeFeedbackModel(), GetParam());
      auto second_session = Session(second_loop ? stateful::MakeFeedbackModel() : stateful::MakeRenamedStageModel(),
                                    GetParam());
      FeedbackConfig first_config, second_config;
      second_config.config.finite_input_names = nullptr;
      second_config.config.num_finite_inputs = 0;
      second_config.config.nonempty_input = nullptr;
      second_config.config.exit_conditions = &second_config.exits[1];
      second_config.config.num_exit_conditions = 1;
      second_config.exits[1].limit = 2;
      auto first = Loop(first_session, first_config.config);
      Handle<OrtSessionLoop> second(nullptr, Api().ReleaseSessionLoop);
      if (second_loop) {
        second = Loop(second_session, second_config.config);
      }
      const OrtStatefulValueSource prompt_source{
          first_loop ? ORT_STATEFUL_EXTERNAL_SEQUENCE : ORT_STATEFUL_EXTERNAL_INPUT,
          first_loop ? "prompt" : "one", 0};
      std::array<OrtStatefulInputBinding, 2> first_inputs{
          OrtStatefulInputBinding{"input_value", prompt_source},
          OrtStatefulInputBinding{"past_value", {ORT_STATEFUL_STATE, "carry", 0}},
      };
      const OrtStatefulStateBinding write_state{"carry", "present_value"};
      std::array<OrtStatefulInputBinding, 2> second_inputs{
          OrtStatefulInputBinding{second_loop ? "past_value" : "upstream",
                                  {ORT_STATEFUL_COMPONENT_OUTPUT, "present_value", 0}},
          OrtStatefulInputBinding{"input_value", {ORT_STATEFUL_EXTERNAL_INPUT, "delta", 0}},
      };
      std::array<OrtStatefulChainComponent, 2> components{};
      components[0].name = "first";
      components[0].session = first_loop ? nullptr : static_cast<OrtSession*>(first_session);
      components[0].loop = first_loop ? first.get() : nullptr;
      components[0].inputs = first_inputs.data();
      components[0].num_inputs = 2;
      components[0].state_outputs = &write_state;
      components[0].num_state_outputs = 1;
      components[1].name = "second";
      components[1].session = second_loop ? nullptr : static_cast<OrtSession*>(second_session);
      components[1].loop = second_loop ? second.get() : nullptr;
      components[1].inputs = second_inputs.data();
      components[1].num_inputs = second_loop ? 2 : 1;
      std::array<OrtStatefulChainOutput, 2> selected{
          OrtStatefulChainOutput{"early", 0, "present_value", ORT_STATEFUL_MODEL_OUTPUT},
          OrtStatefulChainOutput{
              "result",
              1,
              second_loop ? "present_value" : "pipeline_result",
              ORT_STATEFUL_MODEL_OUTPUT,
          },
      };
      OrtStatefulChainConfig config{};
      config.struct_size = sizeof(config);
      config.components = components.data();
      config.num_components = components.size();
      config.outputs = selected.data();
      config.num_outputs = selected.size();
      OrtSessionChainOptions* raw_options = nullptr;
      Ok(Api().CreateSessionChainOptions(&config, &raw_options));
      auto options = Own(raw_options, Api().ReleaseSessionChainOptions);
      Ok(Api().SessionChainOptionsSetConfig(options.get(), &config));
      auto invalid = config;
      invalid.struct_size = 0;
      Error(Api().SessionChainOptionsSetConfig(options.get(), &invalid));
      first.reset();
      second.reset();
      OrtSessionChain* raw_chain = nullptr;
      Ok(Api().CreateSessionChain(options.get(), &raw_chain));
      auto chain = Own(raw_chain, Api().ReleaseSessionChain);
      options.reset();
      auto one = Tensor<float>({}, {2.f}), two = Tensor<float>({}, {3.f});
      auto carry = Tensor<float>({}, {1.f}), delta = Tensor<float>({}, {2.f});
      std::array<OrtStatefulNamedValue, 2> external{
          OrtStatefulNamedValue{"one", one}, OrtStatefulNamedValue{"delta", delta}};
      const OrtValue* prompt[2]{one, two};
      const OrtStatefulInputSequence sequence{"prompt", prompt, 2};
      const OrtStatefulNamedValue initial{"carry", carry};
      OrtStatefulInputs inputs{};
      inputs.struct_size = sizeof(inputs);
      inputs.inputs = first_loop ? (second_loop ? &external[1] : nullptr) : external.data();
      inputs.num_inputs = (first_loop ? 0u : 1u) + (second_loop ? 1u : 0u);
      inputs.sequences = first_loop ? &sequence : nullptr;
      inputs.num_sequences = first_loop ? 1 : 0;
      inputs.initial_state = &initial;
      inputs.num_initial_state = 1;
      OrtStatefulExecution* raw_execution = nullptr;
      Ok(Api().CreateChainExecution(chain.get(), &inputs, &raw_execution));
      auto execution = Own(raw_execution, Api().ReleaseStatefulExecution);
      chain.reset();
      OrtStatefulStepResult* raw_result = nullptr;
      Ok(Api().StatefulExecutionRun(execution.get(), &raw_result));
      auto result = Own(raw_result, Api().ReleaseStatefulStepResult);
      const float early = first_loop ? 6.f : 3.f;
      EXPECT_EQ(Float(Output(result.get(), "early")), early);
      EXPECT_EQ(Float(Output(result.get(), "result")), second_loop ? early + 4.f : early * 2.f);
      auto info = Info(result.get());
      EXPECT_EQ(info.kind, ORT_STATEFUL_CHAIN_RESULT);
      EXPECT_EQ(info.components_completed, 2u);
      EXPECT_EQ(std::string_view(info.component_name), "second");
      Ok(Api().StatefulExecutionGetState(execution.get(), &raw_result));
      auto state = Own(raw_result, Api().ReleaseStatefulStepResult);
      EXPECT_EQ(Float(Output(state.get(), "carry", ORT_STATEFUL_STATE_VALUES)), early);
      Ok(Api().StatefulExecutionReset(execution.get()));
      Ok(Api().StatefulExecutionStep(execution.get(), &raw_result));
      auto step = Own(raw_result, Api().ReleaseStatefulStepResult);
      EXPECT_EQ(Info(step.get()).invocations, 1);
    }
  }
}

TEST_P(StatefulApiProviderTest, CancellationFailureInfoAndOutParameters) {
  auto session = Session(stateful::MakeFeedbackModel(), GetParam());
  FeedbackConfig description;
  auto loop = Loop(session, description.config);
  auto initial = Tensor<float>({}, {0.f}), value = Tensor<float>({}, {2.f});
  const OrtStatefulNamedValue seed{"past_value", initial}, input{"input_value", value};
  const OrtStatefulInputBatch batch{&input, 1};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = &seed;
  inputs.num_inputs = 1;
  inputs.batches = &batch;
  inputs.num_batches = 1;
  OrtStatefulExecution* raw = nullptr;
  Ok(Api().CreateLoopExecution(loop.get(), &inputs, &raw));
  auto execution = Own(raw, Api().ReleaseStatefulExecution);
  Ok(Api().StatefulExecutionCancel(execution.get()));
  OrtStatefulStepResult* result = nullptr;
  Error(Api().StatefulExecutionStep(execution.get(), &result));
  EXPECT_EQ(result, nullptr);
  int invalid = 0;
  Ok(Api().StatefulExecutionRequiresReset(execution.get(), &invalid));
  EXPECT_EQ(invalid, 1);
  Error(Api().StatefulExecutionRun(execution.get(), &result));
  Error(Api().StatefulExecutionGetState(execution.get(), &result));
  Ok(Api().StatefulExecutionReset(execution.get()));
  Ok(Api().StatefulExecutionRequiresReset(execution.get(), &invalid));
  EXPECT_EQ(invalid, 0);
  Ok(Api().StatefulExecutionStep(execution.get(), &result));
  auto owned_result = Own(result, Api().ReleaseStatefulStepResult);
  OrtStatefulStepInfo untouched{};
  untouched.iterations = 77;
  Error(Api().StatefulStepResultGetInfo(result, sizeof(untouched) - 1, &untouched));
  EXPECT_EQ(untouched.iterations, 77);
  OrtStatefulNamedValue sentinel{"sentinel", nullptr};
  const OrtStatefulNamedValue* outputs = &sentinel;
  size_t count = 77;
  Error(Api().StatefulStepResultGetOutputs(result, static_cast<OrtStatefulOutputGroup>(99), &outputs, &count));
  EXPECT_EQ(outputs, &sentinel);
  EXPECT_EQ(count, 77u);
  Error(Api().StatefulStepResultGetOutputs(result, ORT_STATEFUL_SELECTED_OUTPUTS, nullptr, &count));
  EXPECT_EQ(count, 77u);
  Error(Api().StatefulStepResultGetOutputs(result, ORT_STATEFUL_SELECTED_OUTPUTS, &outputs, nullptr));
  EXPECT_EQ(outputs, &sentinel);
}

TEST_P(StatefulApiProviderTest, ModelScalarTokenOutputsKeepGenericExitBehavior) {
  auto model = stateful::MakeModel("public_scalar_tokens");
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "input", ONNX_NAMESPACE::TensorProto_DataType_INT64, {});
  stateful::SetTensorType(*graph.add_output(), "output", ONNX_NAMESPACE::TensorProto_DataType_INT64, {});
  stateful::AddNode(graph, "Identity", {"input"}, "output");
  auto session = Session(model, GetParam());
  const char* names[1]{"input"};
  const char* selected[1]{"output"};
  const int64_t eos = 3;
  const OrtStatefulExitCondition exits[2]{
      {ORT_STATEFUL_EOS, 0, nullptr, &eos, 1},
      {ORT_STATEFUL_MAX_GENERATED_TOKENS, 2, nullptr, nullptr, 0}};
  OrtStatefulLoopConfig config{};
  config.struct_size = sizeof(config);
  config.finite_input_names = names;
  config.num_finite_inputs = 1;
  config.output_names = selected;
  config.num_outputs = 1;
  config.exit_conditions = exits;
  config.num_exit_conditions = 2;
  config.generated_token_output = "output";
  auto loop = Loop(session, config);
  auto one = Tensor<int64_t>({}, {1}), three = Tensor<int64_t>({}, {3});
  const OrtStatefulNamedValue values[2]{{"input", one}, {"input", three}};
  const OrtStatefulInputBatch batches[2]{{&values[0], 1}, {&values[1], 1}};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.batches = batches;
  inputs.num_batches = 2;
  OrtStatefulExecution* raw = nullptr;
  Ok(Api().CreateLoopExecution(loop.get(), &inputs, &raw));
  auto execution = Own(raw, Api().ReleaseStatefulExecution);
  OrtStatefulStepResult* result = nullptr;
  Ok(Api().StatefulExecutionRun(execution.get(), &result));
  auto owned = Own(result, Api().ReleaseStatefulStepResult);
  const auto info = Info(result);
  EXPECT_EQ(info.generated_tokens, 2);
  EXPECT_EQ(info.has_generated_token, 1);
  EXPECT_EQ(info.generated_token, 3);
  EXPECT_EQ(info.num_exit_reasons, 2u);
}

TEST_F(StatefulApiTest, StandaloneCacheSamplerAndExceptionContainment) {
  const int64_t shape[4]{1, 1, 0, 1};
  OrtStatefulKVCacheSpec spec{"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 4, 2, 1};
  OrtStatefulKVCacheConfig config{sizeof(config), &spec, 1};
  OrtKVCacheOptions* raw_options = nullptr;
  Ok(Api().CreateKVCacheOptions(&config, &raw_options));
  auto options = Own(raw_options, Api().ReleaseKVCacheOptions);
  Ok(Api().KVCacheOptionsSetConfig(options.get(), &config));
  auto invalid = config;
  invalid.struct_size = 0;
  Error(Api().KVCacheOptionsSetConfig(options.get(), &invalid));
  Ort::AllocatorWithDefaultOptions allocator;
  OrtKVCacheManager* raw_cache = nullptr;
  Ok(Api().CreateKVCacheManager(options.get(), allocator, &raw_cache));
  auto cache = Own(raw_cache, Api().ReleaseKVCacheManager);
  options.reset();
  auto first = Tensor<float>({1, 1, 1, 1}, {3.f});
  Ok(Api().KVCacheManagerUpdate(cache.get(), "key", first, ORT_KV_CACHE_SHARE));
  OrtKVCacheView* raw_view = nullptr;
  Ok(Api().KVCacheManagerGetView(cache.get(), "key", &raw_view));
  auto view = Own(raw_view, Api().ReleaseKVCacheView);
  const OrtValue* borrowed = nullptr;
  int64_t length = -1, capacity = -1;
  Ok(Api().KVCacheViewGetInfo(view.get(), &borrowed, &length, &capacity));
  EXPECT_EQ(length, 1);
  EXPECT_EQ(capacity, 1);
  EXPECT_EQ(Float(borrowed), 3.f);
  Ok(Api().KVCacheManagerReserve(cache.get(), "key", 4));
  auto delta = Tensor<float>({1, 1, 1, 1}, {4.f});
  Ok(Api().KVCacheManagerUpdate(cache.get(), "key", delta, ORT_KV_CACHE_APPEND));
  auto full = Tensor<float>({1, 1, 2, 1}, {5.f, 6.f});
  Ok(Api().KVCacheManagerUpdate(cache.get(), "key", full, ORT_KV_CACHE_COPY));
  OrtStatefulKVCacheStats stats{};
  Ok(Api().KVCacheManagerGetStats(cache.get(), sizeof(stats), &stats));
  EXPECT_GE(stats.allocations, 3u);
  EXPECT_EQ(stats.shared_updates, 1u);
  EXPECT_GT(stats.bytes_copied, 0u);
  stats.allocations = 77;
  Error(Api().KVCacheManagerGetStats(cache.get(), sizeof(stats) - 1, &stats));
  EXPECT_EQ(stats.allocations, 77u);
  // Overflow throws inside the native manager; no C++ exception may escape the C ABI.
  Error(Api().KVCacheManagerReserve(cache.get(), "key", std::numeric_limits<int64_t>::max()));
  Error(Api().KVCacheManagerUpdate(cache.get(), "key", delta, static_cast<OrtKVCacheUpdateKind>(99)));
  Error(Api().KVCacheManagerGetView(cache.get(), "missing", &raw_view));
  EXPECT_EQ(raw_view, view.get());
  Ok(Api().KVCacheManagerReset(cache.get()));
  cache.reset();
  first = Ort::Value(nullptr);
  EXPECT_EQ(Float(borrowed), 3.f);
  int64_t untouched_length = 77, untouched_capacity = 88;
  const OrtValue* untouched_value = borrowed;
  Error(Api().KVCacheViewGetInfo(nullptr, &untouched_value, &untouched_length, &untouched_capacity));
  EXPECT_EQ(untouched_value, borrowed);
  EXPECT_EQ(untouched_length, 77);
  EXPECT_EQ(untouched_capacity, 88);

  OrtStatefulSamplerConfig sampler{sizeof(sampler), ORT_STATEFUL_GREEDY, 1, 0};
  OrtSamplerOptions* raw_sampler = nullptr;
  Ok(Api().CreateSamplerOptions(&sampler, &raw_sampler));
  auto sampling = Own(raw_sampler, Api().ReleaseSamplerOptions);
  auto logits = Tensor<float>({1, 2, 3}, {1.f, 5.f, 2.f, 9.f, 1.f, 0.f});
  int64_t token = 77;
  Ok(Api().SamplerSample(sampling.get(), logits, &token));
  EXPECT_EQ(token, 1);
  sampler.has_logits_position = 0;
  Ok(Api().SamplerOptionsSetConfig(sampling.get(), &sampler));
  Ok(Api().SamplerSample(sampling.get(), logits, &token));
  EXPECT_EQ(token, 0);
  sampler.has_logits_position = 1;
  sampler.logits_position = -1;
  Error(Api().SamplerOptionsSetConfig(sampling.get(), &sampler));
  Ok(Api().SamplerSample(sampling.get(), logits, &token));
  EXPECT_EQ(token, 0);
  sampler.algorithm = static_cast<OrtStatefulSamplingAlgorithm>(99);
  Error(Api().SamplerOptionsSetConfig(sampling.get(), &sampler));
  sampler.algorithm = ORT_STATEFUL_GREEDY;
  sampler.has_logits_position = 2;
  Error(Api().SamplerOptionsSetConfig(sampling.get(), &sampler));
  token = 77;
  auto nan = Tensor<float>({1, 1}, {std::numeric_limits<float>::quiet_NaN()});
  Error(Api().SamplerSample(sampling.get(), nan, &token));
  EXPECT_EQ(token, 77);
  float storage = 1.f;
  Ort::MemoryInfo gpu("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto device = Ort::Value::CreateTensor<float>(gpu, &storage, 1, std::array<int64_t, 2>{1, 1}.data(), 2);
  Error(Api().SamplerSample(sampling.get(), device, &token));
  EXPECT_EQ(token, 77);
}

TEST_F(StatefulApiTest, PointerSizeEnumAndAtomicConfigurationValidation) {
  FeedbackConfig description;
  struct ExtendedConfig {
    OrtStatefulLoopConfig value;
    uint64_t tail;
  } extended{description.config, 0x12345678};
  extended.value.struct_size = sizeof(extended);
  auto extended_options = LoopOptions(extended.value);
  auto options = LoopOptions(description.config);
  auto invalid = description.config;
  invalid.num_finite_inputs = 1;
  invalid.finite_input_names = nullptr;
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  invalid = description.config;
  const char* empty[1]{""};
  invalid.finite_input_names = empty;
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  invalid = description.config;
  invalid.num_finite_inputs = std::numeric_limits<size_t>::max();
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  invalid = description.config;
  auto exits = description.exits[0];
  invalid.exit_conditions = &exits;
  invalid.num_exit_conditions = 1;
  exits.kind = static_cast<OrtStatefulExitKind>(99);
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  exits = {ORT_STATEFUL_END_OF_INPUTS, 1, nullptr, nullptr, 0};
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  exits = {ORT_STATEFUL_MAX_ITERATIONS, -1, nullptr, nullptr, 0};
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  exits = {ORT_STATEFUL_END_OF_INPUTS, 0, "unused", nullptr, 0};
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  const int64_t eos = 3;
  exits = {ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, &eos, 1};
  Error(Api().SessionLoopOptionsSetConfig(options.get(), &invalid));
  auto session = Session(stateful::MakeFeedbackModel());
  OrtSessionLoop* raw_loop = nullptr;
  Ok(Api().CreateSessionLoop(session, options.get(), &raw_loop));
  auto loop = Own(raw_loop, Api().ReleaseSessionLoop);
  auto value = Tensor<float>({}, {1.f});
  OrtStatefulNamedValue inputs_named[2]{{"past_value", value}, {"past_value", value}};
  OrtStatefulInputs inputs{};
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = inputs_named;
  inputs.num_inputs = 2;
  OrtStatefulExecution* execution = nullptr;
  Error(Api().CreateLoopExecution(loop.get(), &inputs, &execution));
  EXPECT_EQ(execution, nullptr);
  inputs.num_inputs = 1;
  inputs_named[0].value = nullptr;
  Error(Api().CreateLoopExecution(loop.get(), &inputs, &execution));
  inputs_named[0].value = value;
  inputs.num_sequences = 1;
  Error(Api().CreateLoopExecution(loop.get(), &inputs, &execution));
}

ONNX_NAMESPACE::ModelProto PublicGenerationModel() {
  auto model = stateful::MakeDecoderModel();
  auto& graph = *model.mutable_graph();
  graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  stateful::SetTensorType(*graph.mutable_input(0), "token", ONNX_NAMESPACE::TensorProto_DataType_INT64, {1, -1});
  graph.mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  stateful::SetTensorType(*graph.mutable_output(0), "logits", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, -1, 4});
  for (auto& initializer : *graph.mutable_initializer()) {
    if (initializer.name() == "key_shape") {
      initializer.clear_int64_data();
      for (int64_t value : {1, 1, -1, 1}) {
        initializer.add_int64_data(value);
      }
    }
  }
  for (const auto& port : {std::pair{"mask", "mask_seen"}, std::pair{"positions", "positions_seen"}}) {
    stateful::SetTensorType(*graph.add_input(), port.first, ONNX_NAMESPACE::TensorProto_DataType_INT64, {1, -1});
    stateful::SetTensorType(*graph.add_output(), port.second, ONNX_NAMESPACE::TensorProto_DataType_INT64, {1, -1});
    stateful::AddNode(graph, "Identity", {port.first}, port.second);
  }
  return model;
}

TEST_P(StatefulApiProviderTest, NativeGenerationCachesAndZeroBudgetThroughPublicC) {
  for (int64_t budget : {0, 2}) {
    auto session = Session(PublicGenerationModel(), GetParam());
    const int64_t dimensions[4]{1, 1, 0, 1};
    OrtStatefulKVCacheBinding cache{
        {"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dimensions, 4, 2, 0},
        "past_key",
        "present_key",
        ORT_KV_CACHE_SHARE};
    const char* prompt_names[1]{"token"};
    const char* outputs[1]{"logits"};
    const int64_t eos = 2;
    OrtStatefulExitCondition prefill_exits[2]{
        {ORT_STATEFUL_END_OF_INPUTS, 0, nullptr, nullptr, 0},
        {ORT_STATEFUL_MAX_GENERATED_TOKENS, budget, nullptr, nullptr, 0},
    };
    OrtStatefulExitCondition decode_exits[2]{
        {ORT_STATEFUL_MAX_GENERATED_TOKENS, budget, nullptr, nullptr, 0},
        {ORT_STATEFUL_EOS, 0, nullptr, &eos, 1},
    };
    OrtStatefulGenerationConfig generation{
        sizeof(generation), ORT_STATEFUL_PREFILL, "token", "logits", "mask", "positions", 1, 0};
    OrtStatefulLoopConfig config{};
    config.struct_size = sizeof(config);
    config.finite_input_names = prompt_names;
    config.num_finite_inputs = 1;
    config.output_names = outputs;
    config.num_outputs = 1;
    config.exit_conditions = prefill_exits;
    config.num_exit_conditions = 2;
    config.kv_caches = &cache;
    config.num_kv_caches = 1;
    config.generation = &generation;
    auto prefill = Loop(session, config);
    auto prompt = Tensor<int64_t>({1, 1}, {0});
    const OrtStatefulNamedValue prompt_value{"token", prompt};
    const OrtStatefulInputBatch batch{&prompt_value, 1};
    OrtStatefulInputs loop_inputs{};
    loop_inputs.struct_size = sizeof(loop_inputs);
    loop_inputs.batches = &batch;
    loop_inputs.num_batches = 1;
    OrtStatefulExecution* raw_prefill = nullptr;
    Ok(Api().CreateLoopExecution(prefill.get(), &loop_inputs, &raw_prefill));
    auto prefill_execution = Own(raw_prefill, Api().ReleaseStatefulExecution);
    OrtStatefulStepResult* prefill_result = nullptr;
    Ok(Api().StatefulExecutionRun(prefill_execution.get(), &prefill_result));
    auto retained_prefill = Own(prefill_result, Api().ReleaseStatefulStepResult);
    Ok(Api().StatefulExecutionGetState(prefill_execution.get(), &prefill_result));
    auto prefill_state = Own(prefill_result, Api().ReleaseStatefulStepResult);
    Ort::ConstValue prefill_keys(Output(prefill_state.get(), "key", ORT_STATEFUL_STATE_VALUES));
    EXPECT_EQ(prefill_keys.GetTensorTypeAndShapeInfo().GetShape()[2], budget == 0 ? 0 : 1);
    Ok(Api().StatefulExecutionReset(prefill_execution.get()));
    EXPECT_EQ(prefill_keys.GetTensorTypeAndShapeInfo().GetShape()[2], budget == 0 ? 0 : 1);
    generation.phase = ORT_STATEFUL_DECODE;
    generation.has_logits_position = 0;
    config.finite_input_names = nullptr;
    config.num_finite_inputs = 0;
    config.exit_conditions = decode_exits;
    auto decode = Loop(session, config);
    const OrtStatefulInputBinding prompt_binding{"token", {ORT_STATEFUL_EXTERNAL_INPUT, "prompt", 0}};
    OrtStatefulChainComponent components[2]{};
    components[0].name = "prefill";
    components[0].loop = prefill.get();
    components[0].inputs = &prompt_binding;
    components[0].num_inputs = 1;
    components[1].name = "decode";
    components[1].loop = decode.get();
    const OrtStatefulChainOutput selected{"tokens", 0, nullptr, ORT_STATEFUL_GENERATED_TOKEN_IDS};
    OrtStatefulChainConfig chain_config{sizeof(chain_config), components, 2, &selected, 1};
    OrtSessionChainOptions* raw_options = nullptr;
    Ok(Api().CreateSessionChainOptions(&chain_config, &raw_options));
    auto options = Own(raw_options, Api().ReleaseSessionChainOptions);
    OrtSessionChain* raw_chain = nullptr;
    Ok(Api().CreateSessionChain(options.get(), &raw_chain));
    auto chain = Own(raw_chain, Api().ReleaseSessionChain);
    const OrtStatefulNamedValue input{"prompt", prompt};
    OrtStatefulInputs inputs{};
    inputs.struct_size = sizeof(inputs);
    inputs.inputs = &input;
    inputs.num_inputs = 1;
    OrtStatefulExecution* raw_execution = nullptr;
    Ok(Api().CreateChainExecution(chain.get(), &inputs, &raw_execution));
    auto execution = Own(raw_execution, Api().ReleaseStatefulExecution);
    OrtStatefulStepResult* raw_result = nullptr;
    Ok(Api().StatefulExecutionRun(execution.get(), &raw_result));
    auto result = Own(raw_result, Api().ReleaseStatefulStepResult);
    EXPECT_EQ(Info(result.get()).generated_tokens, budget);
    Ort::ConstValue tokens(Output(result.get(), "tokens"));
    EXPECT_EQ(tokens.GetTensorTypeAndShapeInfo().GetElementCount(), static_cast<size_t>(budget));
    if (budget != 0) {
      EXPECT_EQ(tokens.GetTensorData<int64_t>()[0], 1);
      EXPECT_EQ(tokens.GetTensorData<int64_t>()[1], 2);
    }
    Ok(Api().StatefulExecutionGetState(execution.get(), &raw_result));
    auto state = Own(raw_result, Api().ReleaseStatefulStepResult);
    Ort::ConstValue keys(Output(state.get(), "key", ORT_STATEFUL_STATE_VALUES));
    EXPECT_EQ(keys.GetTensorTypeAndShapeInfo().GetShape()[2], budget);
    Ok(Api().StatefulExecutionReset(execution.get()));
    Ok(Api().StatefulExecutionCancel(execution.get()));
    Error(Api().StatefulExecutionStep(execution.get(), &raw_result));
    Ok(Api().StatefulExecutionReset(execution.get()));
  }
}

TEST_F(StatefulApiTest, EveryStatusEntryValidatesRequiredPointers) {
  const auto& api = Api();
  Error(api.CreateSessionLoopOptions(nullptr, nullptr));
  OrtSessionLoopOptions* options = nullptr;
  Error(api.CreateSessionLoopOptions(nullptr, &options));
  EXPECT_EQ(options, nullptr);
  Error(api.SessionLoopOptionsSetConfig(nullptr, nullptr));
  Error(api.CreateSessionLoop(nullptr, nullptr, nullptr));
  Error(api.CreateSessionChainOptions(nullptr, nullptr));
  Error(api.SessionChainOptionsSetConfig(nullptr, nullptr));
  Error(api.CreateSessionChain(nullptr, nullptr));
  Error(api.CreateLoopExecution(nullptr, nullptr, nullptr));
  Error(api.CreateChainExecution(nullptr, nullptr, nullptr));
  Error(api.StatefulExecutionStep(nullptr, nullptr));
  Error(api.StatefulExecutionRun(nullptr, nullptr));
  Error(api.StatefulExecutionReset(nullptr));
  Error(api.StatefulExecutionCancel(nullptr));
  Error(api.StatefulExecutionRequiresReset(nullptr, nullptr));
  Error(api.StatefulExecutionGetState(nullptr, nullptr));
  Error(api.StatefulStepResultGetInfo(nullptr, 0, nullptr));
  Error(api.StatefulStepResultGetOutputs(nullptr, ORT_STATEFUL_SELECTED_OUTPUTS, nullptr, nullptr));
  Error(api.CreateKVCacheOptions(nullptr, nullptr));
  Error(api.KVCacheOptionsSetConfig(nullptr, nullptr));
  Error(api.CreateKVCacheManager(nullptr, nullptr, nullptr));
  Error(api.KVCacheManagerGetView(nullptr, nullptr, nullptr));
  Error(api.KVCacheViewGetInfo(nullptr, nullptr, nullptr, nullptr));
  Error(api.KVCacheManagerUpdate(nullptr, nullptr, nullptr, ORT_KV_CACHE_COPY));
  Error(api.KVCacheManagerReserve(nullptr, nullptr, 0));
  Error(api.KVCacheManagerReset(nullptr));
  Error(api.KVCacheManagerGetStats(nullptr, 0, nullptr));
  Error(api.CreateSamplerOptions(nullptr, nullptr));
  Error(api.SamplerOptionsSetConfig(nullptr, nullptr));
  Error(api.SamplerSample(nullptr, nullptr, nullptr));
}

INSTANTIATE_TEST_SUITE_P(Cpu, StatefulApiProviderTest, testing::Values(false));
#ifdef USE_CUDA
INSTANTIATE_TEST_SUITE_P(Cuda, StatefulApiProviderTest, testing::Values(true));
#endif

}  // namespace
}  // namespace onnxruntime::test
