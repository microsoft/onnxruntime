// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <functional>
#include <future>
#include <limits>

#include "gtest/gtest.h"
#include "core/graph/graph_viewer.h"
#include "core/session/inference_session.h"
#include "core/session/IOBinding.h"
#include "core/session/session_chain.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/stateful_execution_test_utils.h"

namespace onnxruntime::test {
namespace {

class ChainTestSession : public InferenceSession {
 public:
  using InferenceSession::InferenceSession;
  Status Run(const RunOptions& options, IOBinding& binding) override {
    ++calls;
    if (before_run) {
      ORT_RETURN_IF_ERROR(before_run());
    }
    ORT_RETURN_IF_ERROR(InferenceSession::Run(options, binding));
    return after_run ? after_run(binding) : Status::OK();
  }
  size_t calls = 0;
  std::function<Status()> before_run;
  std::function<Status(IOBinding&)> after_run;
};

ChainValueSource External(const char* name) { return {ChainSourceKind::ExternalInput, name}; }
ChainValueSource Sequence(const char* name) { return {ChainSourceKind::ExternalSequence, name}; }
ChainValueSource Output(size_t index, const char* name) { return {ChainSourceKind::ComponentOutput, name, index}; }
ChainValueSource State(const char* name) { return {ChainSourceKind::State, name}; }

class SessionChainTest : public testing::TestWithParam<bool> {
 protected:
  std::unique_ptr<ChainTestSession> Session(ONNX_NAMESPACE::ModelProto model, bool cuda) {
    SessionOptions options;
    options.intra_op_param.thread_pool_size = 1;
    options.inter_op_param.thread_pool_size = 1;
    options.graph_optimization_level = TransformerLevel::Default;
    auto session = std::make_unique<ChainTestSession>(options, GetEnvironment());
    if (cuda) {
      auto provider = DefaultCudaExecutionProvider();
      EXPECT_NE(provider, nullptr);
      ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(std::move(provider)));
    }
    const auto bytes = model.SerializeAsString();
    ORT_THROW_IF_ERROR(session->Load(bytes.data(), static_cast<int>(bytes.size())));
    ORT_THROW_IF_ERROR(session->Initialize());
    if (cuda) {
      for (const auto& node : session->GetSessionState().GetGraphViewer().Nodes()) {
        EXPECT_EQ(node.GetExecutionProviderType(), kCudaExecutionProvider);
      }
    }
    return session;
  }

  std::unique_ptr<ChainTestSession> Session(ONNX_NAMESPACE::ModelProto model) {
    return Session(std::move(model), GetParam());
  }

  template <typename T>
  OrtValue Value(std::initializer_list<int64_t> shape, std::initializer_list<T> values) {
    OrtValue value;
    CreateMLValue<T>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], shape, values, &value);
    return value;
  }

  OrtValue Scalar(float value) { return Value<float>({}, {value}); }

  std::shared_ptr<const SessionLoop> Loop(InferenceSession& session, SessionLoopOptions options) {
    std::shared_ptr<const SessionLoop> loop;
    ORT_THROW_IF_ERROR(SessionLoop::Create(session, std::move(options), loop));
    return loop;
  }

  SessionLoopOptions FiniteLoopOptions() {
    SessionLoopOptions options;
    options.finite_input_names = {"input_value"};
    options.feedback = {{"past_value", "present_value"}};
    options.output_names = {"done"};
    options.exit_conditions = {{LoopExitKind::EndOfInputs}};
    return options;
  }

  SessionChainOptions Single(InferenceSession& session) {
    return {{{"double", &session, {{"upstream", External("value")}}, {}}},
            {{"result", 0, "pipeline_result"}}};
  }

  std::unique_ptr<SessionChainExecution> Execution(SessionChainOptions options, SessionChainInputs inputs) {
    std::shared_ptr<const SessionChain> chain;
    ORT_THROW_IF_ERROR(SessionChain::Create(std::move(options), chain));
    std::unique_ptr<SessionChainExecution> execution;
    ORT_THROW_IF_ERROR(chain->CreateExecution(std::move(inputs), execution));
    return execution;
  }
};

TEST_P(SessionChainTest, AllSessionAndLoopCombinationsHaveReferenceResults) {
  for (bool first_is_loop : {false, true}) {
    for (bool second_is_loop : {false, true}) {
      auto first = Session(stateful::MakeFeedbackModel());
      auto second = Session(second_is_loop ? stateful::MakeFeedbackModel() : stateful::MakeRenamedStageModel());
      SessionChainComponent a{
          "first", first.get(), {{"input_value", External("input")}, {"past_value", External("initial")}}, {}};
      SessionChainComponent b{"second", second.get(), {{"upstream", Output(0, "present_value")}}, {}};
      SessionChainInputs inputs{{{"input", Scalar(2.f)}, {"initial", Scalar(1.f)}}};
      if (first_is_loop) {
        a.execution = Loop(*first, FiniteLoopOptions());
        a.inputs[0].source = Sequence("prompt");
        inputs.inputs.erase("input");
        inputs.sequences.emplace("prompt", InlinedVector<OrtValue>{Scalar(2.f), Scalar(3.f), Scalar(4.f)});
      }
      if (second_is_loop) {
        auto options = FiniteLoopOptions();
        options.finite_input_names.clear();
        options.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
        b.execution = Loop(*second, options);
        b.inputs = {{"input_value", External("delta")}, {"past_value", Output(0, "present_value")}};
        inputs.inputs.emplace("delta", Scalar(2.f));
      }
      SessionChainOptions options{
          {a, b},
          {{"first_result", 0, "present_value"},
           {"first_done", 0, "done"},
           {"result", 1, second_is_loop ? "present_value" : "pipeline_result"}}};
      auto execution = Execution(options, inputs);
      SessionChainStepResult result;
      ASSERT_STATUS_OK(execution->Run(result));
      const float first_expected = first_is_loop ? 10.f : 3.f;
      const float expected = second_is_loop ? first_expected + 4.f : first_expected * 2.f;
      EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), expected);
      EXPECT_EQ(*result.outputs.at("first_result").Get<Tensor>().Data<float>(), first_expected);
      EXPECT_EQ(*result.outputs.at("first_done").Get<Tensor>().Data<bool>(), first_is_loop);
      EXPECT_EQ(result.invocations, (first_is_loop ? 3 : 1) + (second_is_loop ? 2 : 1));
      EXPECT_EQ(result.components_completed, 2u);
      EXPECT_EQ(result.component_name, "second");
      EXPECT_EQ(first->calls, first_is_loop ? 3u : 1u);
      EXPECT_EQ(second->calls, second_is_loop ? 2u : 1u);
    }
  }
}

TEST_P(SessionChainTest, RetainedStateStepRunIsolationAndReset) {
  auto first = Session(stateful::MakeFeedbackModel());
  auto second = Session(stateful::MakeFeedbackModel());
  auto loop = Loop(*first, FiniteLoopOptions());
  SessionChainComponent prefill{
      "prefill",
      loop,
      {{"input_value", Sequence("prompt")}, {"past_value", State("cache")}},
      {{"cache", "present_value"}},
  };
  SessionChainComponent decode{
      "decode",
      second.get(),
      {{"input_value", External("delta")}, {"past_value", State("cache")}},
      {{"cache", "present_value"}},
  };
  SessionChainOptions options{{prefill, decode},
                              {{"result", 1, "present_value"}, {"prefill_result", 0, "present_value"}}};
  SessionChainInputs inputs{{{"delta", Scalar(4.f)}},
                            {{"prompt", {Scalar(2.f), Scalar(3.f)}}},
                            {{"cache", Scalar(0.f)}}};
  std::shared_ptr<const SessionChain> chain;
  ASSERT_STATUS_OK(SessionChain::Create(options, chain));
  std::unique_ptr<SessionChainExecution> execution, independent;
  ASSERT_STATUS_OK(chain->CreateExecution(inputs, execution));
  ASSERT_STATUS_OK(chain->CreateExecution(inputs, independent));
  chain.reset();
  loop.reset();
  options = {};
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_EQ(result.invocations, 1);
  EXPECT_EQ(result.components_completed, 0u);
  EXPECT_EQ(result.component_name, "prefill");
  EXPECT_TRUE(result.outputs.empty());
  EXPECT_EQ(second->calls, 0u);
  EXPECT_EQ(*execution->State().at("cache").Get<Tensor>().Data<float>(), 2.f);
  auto first_result = result;
  auto first_state = execution->State();
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_EQ(result.components_completed, 1u);
  EXPECT_EQ(second->calls, 0u);
  EXPECT_EQ(*execution->State().at("cache").Get<Tensor>().Data<float>(), 5.f);
  EXPECT_EQ(*independent->State().at("cache").Get<Tensor>().Data<float>(), 0.f);
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_TRUE(result.completed);
  EXPECT_EQ(result.invocations, 3);
  EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), 9.f);
  auto completed = result;
  const auto calls = first->calls + second->calls;
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_FALSE(result.executed);
  EXPECT_TRUE(result.outputs.empty());
  EXPECT_TRUE(result.component_outputs.empty());
  EXPECT_EQ(result.invocations, 3);
  EXPECT_EQ(first->calls + second->calls, calls);
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_FALSE(result.executed);
  execution->Reset();
  EXPECT_EQ(*execution->State().at("cache").Get<Tensor>().Data<float>(), 0.f);
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), 9.f);
  ASSERT_STATUS_OK(independent->Run(result));
  EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), 9.f);
  EXPECT_EQ(*first_result.component_outputs.at("present_value").Get<Tensor>().Data<float>(), 2.f);
  EXPECT_EQ(*first_state.at("cache").Get<Tensor>().Data<float>(), 2.f);
  EXPECT_EQ(*completed.outputs.at("result").Get<Tensor>().Data<float>(), 9.f);
}

TEST_P(SessionChainTest, SingleFiniteBatchAndMultipleFiniteInputs) {
  auto session = Session(stateful::MakeFeedbackModel());
  auto loop = Loop(*session, FiniteLoopOptions());
  SessionChainOptions options{
      {{"prefill", loop, {{"input_value", External("prompt")}, {"past_value", External("initial")}}, {}}},
      {{"result", 0, "present_value"}}};
  auto execution = Execution(options, {{{"prompt", Scalar(3.f)}, {"initial", Scalar(2.f)}}});
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.invocations, 1);
  EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), 5.f);

  auto loop_options = FiniteLoopOptions();
  loop_options.feedback.clear();
  loop_options.finite_input_names.push_back("past_value");
  loop = Loop(*session, loop_options);
  options.components[0].execution = loop;
  options.components[0].inputs = {{"input_value", Sequence("values")}, {"past_value", Sequence("past")}};
  SessionChainInputs inputs{{}, {{"values", {Scalar(1.f), Scalar(2.f)}}, {"past", {Scalar(3.f), Scalar(4.f)}}}};
  execution = Execution(options, inputs);
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.invocations, 2);
  EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), 6.f);
}

TEST_P(SessionChainTest, ZeroIterationStagesDoNotInvokeOrInventOutputs) {
  auto first = Session(stateful::MakeRenamedStageModel());
  auto second = Session(stateful::MakeRenamedStageModel());
  SessionLoopOptions options;
  options.output_names = {"pipeline_result"};
  options.exit_conditions = {{LoopExitKind::MaxIterations, 0}};
  auto loop = Loop(*first, options);
  SessionChainOptions chain{
      {{"skip", loop, {{"upstream", External("value")}}, {}},
       {"run", second.get(), {{"upstream", External("value")}}, {}}},
      {{"result", 1, "pipeline_result"}}};
  auto execution = Execution(chain, {{{"value", Scalar(3.f)}}});
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_TRUE(result.completed);
  EXPECT_TRUE(result.executed);
  EXPECT_EQ(result.invocations, 1);
  EXPECT_EQ(result.components_completed, 2u);
  EXPECT_EQ(first->calls, 0u);
  EXPECT_EQ(second->calls, 1u);
  EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), 6.f);

  chain.components[1].inputs[0].source = Output(0, "pipeline_result");
  execution = Execution(chain, {{{"value", Scalar(3.f)}}});
  result.invocations = 99;
  const auto status = execution->Step(result);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("run"), std::string::npos);
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(result.invocations, 99);
  EXPECT_EQ(second->calls, 1u);

  chain.components[0].state_outputs = {{"missing_cache", "pipeline_result"}};
  chain.components[1].inputs[0].source = State("missing_cache");
  execution = Execution(chain, {{{"value", Scalar(3.f)}}});
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());

  chain.components.pop_back();
  chain.outputs = {{"result", 0, "pipeline_result"}};
  execution = Execution(chain, {{{"value", Scalar(3.f)}}});
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
}

TEST_P(SessionChainTest, TrailingZeroIterationLoopCanCompleteWithoutAnotherInvocation) {
  auto session = Session(stateful::MakeRenamedStageModel());
  SessionLoopOptions loop_options;
  loop_options.output_names = {"pipeline_result"};
  loop_options.exit_conditions = {{LoopExitKind::MaxIterations, 0}};
  auto loop = Loop(*session, loop_options);
  auto options = Single(*session);
  options.components.push_back({"skip", loop, {{"upstream", External("value")}}, {}});
  auto execution = Execution(options, {{{"value", Scalar(2.f)}}});
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_FALSE(result.completed);
  EXPECT_TRUE(result.executed);
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_TRUE(result.completed);
  EXPECT_FALSE(result.executed);
  EXPECT_EQ(result.component_name, "skip");
  EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), 4.f);
  EXPECT_EQ(session->calls, 1u);
}

TEST_P(SessionChainTest, FailedComponentStopsDownstreamAndRequiresReset) {
  auto first = Session(stateful::MakeRenamedStageModel());
  auto failing = Session(stateful::MakeRenamedStageModel());
  auto last = Session(stateful::MakeRenamedStageModel());
  auto options = Single(*first);
  options.components.push_back({"failing", failing.get(), {{"upstream", Output(0, "pipeline_result")}}, {}});
  options.components.push_back({"last", last.get(), {{"upstream", Output(1, "pipeline_result")}}, {}});
  options.outputs = {{"result", 2, "pipeline_result"}};
  auto execution = Execution(options, {{{"value", Scalar(1.f)}}});
  failing->before_run = [] { return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "injected failure"); };
  SessionChainStepResult result;
  result.invocations = 99;
  const auto status = execution->Run(result);
  EXPECT_EQ(status.Code(), common::EP_FAIL);
  EXPECT_NE(status.ErrorMessage().find("failing"), std::string::npos);
  EXPECT_NE(status.ErrorMessage().find("injected failure"), std::string::npos);
  EXPECT_EQ(last->calls, 0u);
  EXPECT_EQ(result.invocations, 99);
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_FALSE(execution->Step(result).IsOK());
  failing->before_run = {};
  execution->Reset();
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(*result.outputs.at("result").Get<Tensor>().Data<float>(), 8.f);
  execution->Reset();
  failing->before_run = []() -> Status { ORT_THROW("injected exception"); };
  EXPECT_THROW(ORT_THROW_IF_ERROR(execution->Run(result)), OnnxRuntimeException);
  EXPECT_TRUE(execution->RequiresReset());
}

TEST_P(SessionChainTest, CancellationBeforeAndDuringComponent) {
  auto session = Session(stateful::MakeRenamedStageModel());
  auto execution = Execution(Single(*session), {{{"value", Scalar(2.f)}}});
  execution->Cancel();
  SessionChainStepResult result;
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(session->calls, 0u);
  execution->Reset();
  std::promise<void> entered, resume;
  auto resumed = resume.get_future();
  session->before_run = [&] {
    entered.set_value();
    resumed.wait();
    return Status::OK();
  };
  auto status = std::async(std::launch::async, [&] { return execution->Run(result); });
  entered.get_future().wait();
  execution->Cancel();
  resume.set_value();
  EXPECT_FALSE(status.get().IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  session->before_run = {};
  execution->Reset();
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_TRUE(result.completed);
}

TEST_P(SessionChainTest, DefinitionValidationRejectsInvalidReferencesAndBindings) {
  auto session = Session(stateful::MakeRenamedStageModel());
  const auto valid = Single(*session);
  std::shared_ptr<const SessionChain> chain;
  auto reject = [&](SessionChainOptions options) {
    EXPECT_FALSE(SessionChain::Create(std::move(options), chain).IsOK());
    EXPECT_EQ(chain, nullptr);
  };
  auto options = valid;
  options.components.clear();
  reject(options);
  options = valid;
  options.outputs.clear();
  reject(options);
  options = valid;
  options.components[0].name.clear();
  reject(options);
  options = valid;
  options.components.push_back(options.components[0]);
  reject(options);
  options = valid;
  options.components[0].execution = static_cast<InferenceSession*>(nullptr);
  reject(options);
  options.components[0].execution = std::shared_ptr<const SessionLoop>{};
  reject(options);
  options = valid;
  options.components[0].inputs.clear();
  reject(options);
  options = valid;
  options.components[0].inputs[0].input_name = "missing";
  reject(options);
  options = valid;
  options.components[0].inputs[0].source.name.clear();
  reject(options);
  options = valid;
  options.components[0].inputs[0].source = Sequence("values");
  reject(options);
  options.components[0].inputs[0].source = Output(0, "pipeline_result");
  reject(options);
  options.components[0].inputs[0].source = Output(5, "pipeline_result");
  reject(options);
  options.components[0].inputs[0].source = {static_cast<ChainSourceKind>(99), "value"};
  reject(options);
  options = valid;
  options.components.push_back({"second", session.get(), {{"upstream", Output(0, "missing")}}, {}});
  reject(options);
  for (const auto& binding : {ChainOutputBinding{"", 0, "pipeline_result"},
                              ChainOutputBinding{"x", 10, "pipeline_result"},
                              ChainOutputBinding{"x", 0, "missing"}}) {
    options = valid;
    options.outputs = {binding};
    reject(options);
  }
  options = valid;
  options.outputs.push_back(options.outputs[0]);
  reject(options);
  for (const auto& binding : {ChainStateBinding{"", "pipeline_result"}, ChainStateBinding{"cache", "missing"}}) {
    options = valid;
    options.components[0].state_outputs = {binding};
    reject(options);
  }
  options = valid;
  options.components[0].state_outputs = {{"cache", "pipeline_result"}, {"cache", "pipeline_result"}};
  reject(options);

  auto feedback = Session(stateful::MakeFeedbackModel());
  options = {{{"feedback", feedback.get(), {{"input_value", External("one")}, {"input_value", External("two")}}, {}}},
             {{"x", 0, "present_value"}}};
  reject(options);
}

TEST_P(SessionChainTest, ExternalInputsAndInitialStateAreValidatedBeforeRunning) {
  auto session = Session(stateful::MakeRenamedStageModel());
  std::shared_ptr<const SessionChain> chain;
  ASSERT_STATUS_OK(SessionChain::Create(Single(*session), chain));
  auto reject = [&](SessionChainInputs inputs) {
    std::unique_ptr<SessionChainExecution> execution;
    EXPECT_FALSE(chain->CreateExecution(std::move(inputs), execution).IsOK());
    EXPECT_EQ(execution, nullptr);
  };
  reject({});
  reject({{{"other", Scalar(1.f)}}});
  reject({{{"value", {}}}});
  reject({{{"value", Value<int64_t>({}, {1})}}});
  reject({{{"value", Value<float>({1}, {1.f})}}});
  float storage = 1.f;
  OrtValue device_value;
  OrtMemoryInfo info("UnreachableDevice", OrtAllocatorType::OrtDeviceAllocator,
                     OrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::QUALCOMM, 0));
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{}, &storage, info, device_value);
  reject({{{"value", device_value}}});
  auto options = Single(*session);
  options.components[0].inputs[0].source = State("cache");
  ASSERT_STATUS_OK(SessionChain::Create(options, chain));
  reject({});
  reject({{}, {}, {{"other", Scalar(1.f)}}});
  reject({{}, {}, {{"cache", Value<int64_t>({}, {1})}}});
  EXPECT_EQ(session->calls, 0u);
}

TEST_P(SessionChainTest, FiniteSequencesRejectMissingNamesBadTypesAndUnequalLengths) {
  auto session = Session(stateful::MakeFeedbackModel());
  auto options = FiniteLoopOptions();
  options.feedback.clear();
  options.finite_input_names.push_back("past_value");
  SessionChainOptions chain_options{
      {{"loop", Loop(*session, options), {{"input_value", Sequence("a")}, {"past_value", Sequence("b")}}, {}}},
      {{"result", 0, "present_value"}}};
  std::shared_ptr<const SessionChain> chain;
  ASSERT_STATUS_OK(SessionChain::Create(chain_options, chain));
  auto reject = [&](SessionChainInputs inputs) {
    std::unique_ptr<SessionChainExecution> execution;
    EXPECT_FALSE(chain->CreateExecution(std::move(inputs), execution).IsOK());
  };
  reject({{}, {{"a", {Scalar(1.f)}}, {"unknown", {Scalar(1.f)}}}});
  reject({{}, {{"a", {Scalar(1.f)}}, {"b", {}}}});
  reject({{}, {{"a", {Value<int64_t>({}, {1})}}, {"b", {Scalar(1.f)}}}});
  chain_options.components[0].inputs[1].source = External("one");
  ASSERT_STATUS_OK(SessionChain::Create(chain_options, chain));
  reject({{{"one", Scalar(1.f)}}, {{"a", {Scalar(1.f), Scalar(2.f)}}}});
  EXPECT_EQ(session->calls, 0u);
}

ONNX_NAMESPACE::ModelProto IdentityModel(const char* name, int dtype,
                                         std::initializer_list<int64_t> dimensions) {
  auto model = stateful::MakeModel(name);
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "input", dtype, dimensions);
  stateful::SetTensorType(*graph.add_output(), "output", dtype, dimensions);
  stateful::AddNode(graph, "Identity", {"input"}, "output");
  return model;
}

TEST_P(SessionChainTest, StaticDtypeRankAndDimensionMismatchRejected) {
  auto first = Session(stateful::MakeRenamedStageModel());
  for (const auto& spec : {std::pair<int, InlinedVector<int64_t>>{ONNX_NAMESPACE::TensorProto_DataType_INT64, {}},
                           {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1}}}) {
    auto model = stateful::MakeModel("incompatible_target");
    auto& graph = *model.mutable_graph();
    auto* input = graph.add_input();
    stateful::SetTensorType(*input, "input", spec.first, {});
    for (int64_t dim : spec.second) {
      input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }
    *graph.add_output() = *input;
    graph.mutable_output(0)->set_name("output");
    stateful::AddNode(graph, "Identity", {"input"}, "output");
    auto second = Session(model);
    auto options = Single(*first);
    options.components.push_back({"second", second.get(), {{"input", Output(0, "pipeline_result")}}, {}});
    std::shared_ptr<const SessionChain> chain;
    EXPECT_FALSE(SessionChain::Create(options, chain).IsOK());
    options.components[0].state_outputs = {{"cache", "pipeline_result"}};
    options.components[1].inputs[0].source = State("cache");
    EXPECT_FALSE(SessionChain::Create(options, chain).IsOK());
  }
  auto a = Session(IdentityModel("size_two", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2}));
  auto b = Session(IdentityModel("size_three", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {3}));
  SessionChainOptions options{
      {{"a", a.get(), {{"input", External("value")}}, {{"cache", "output"}}},
       {"b", b.get(), {{"input", Output(0, "output")}}, {}}},
      {{"result", 1, "output"}}};
  std::shared_ptr<const SessionChain> chain;
  EXPECT_FALSE(SessionChain::Create(options, chain).IsOK());
  options.components[1].inputs[0].source = External("other");
  options.components[1].state_outputs = {{"cache", "output"}};
  EXPECT_FALSE(SessionChain::Create(options, chain).IsOK());
}

TEST_P(SessionChainTest, DynamicDimensionMismatchFailsBeforeDownstreamInvocation) {
  auto first = Session(IdentityModel("dynamic", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {-1}));
  auto second = Session(IdentityModel("fixed", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2}));
  SessionChainOptions options{
      {{"source", first.get(), {{"input", External("value")}}, {}},
       {"target", second.get(), {{"input", Output(0, "output")}}, {}}},
      {{"result", 1, "output"}}};
  auto execution = Execution(options, {{{"value", Value<float>({3}, {1.f, 2.f, 3.f})}}});
  SessionChainStepResult result;
  const auto status = execution->Run(result);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("target"), std::string::npos);
  EXPECT_EQ(first->calls, 1u);
  EXPECT_EQ(second->calls, 0u);
  EXPECT_TRUE(execution->RequiresReset());
}

TEST_P(SessionChainTest, CpuCudaHostStagingAndRetainedTensorLifetimes) {
  auto first = Session(stateful::MakeRenamedStageModel(), false);
  auto middle = Session(stateful::MakeRenamedStageModel());
  auto last = Session(stateful::MakeRenamedStageModel(), false);
  auto options = Single(*first);
  options.components.push_back({"device", middle.get(), {{"upstream", Output(0, "pipeline_result")}}, {}});
  options.components.push_back({"host", last.get(), {{"upstream", Output(1, "pipeline_result")}}, {}});
  options.outputs = {{"first", 0, "pipeline_result"}, {"device", 1, "pipeline_result"}, {"last", 2, "pipeline_result"}};
  auto source = Scalar(2.f);
  auto execution = Execution(options, {{{"value", source}}});
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  source.GetMutable<Tensor>()->MutableData<float>()[0] = 9.f;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(*result.outputs.at("first").Get<Tensor>().Data<float>(), 4.f);
  EXPECT_EQ(*result.outputs.at("device").Get<Tensor>().Data<float>(), 8.f);
  EXPECT_EQ(*result.outputs.at("last").Get<Tensor>().Data<float>(), 16.f);
  for (const auto& [name, value] : result.outputs) {
    EXPECT_TRUE(value.Get<Tensor>().Location().device.UsesCpuMemory());
  }
  execution.reset();
  EXPECT_EQ(*result.outputs.at("last").Get<Tensor>().Data<float>(), 16.f);
}

TEST_P(SessionChainTest, DynamicCacheTensorsAndLogitsPassBetweenStages) {
  auto first = Session(stateful::MakeDecoderModel());
  auto second = Session(stateful::MakeDecoderModel());
  SessionChainOptions options{
      {
          {
              "prefill",
              first.get(),
              {{"token", External("first_token")}, {"past_key", External("empty_cache")}},
              {{"kv", "present_key"}},
          },
          {
              "decode",
              second.get(),
              {{"token", External("next_token")}, {"past_key", State("kv")}},
              {{"kv", "present_key"}},
          },
      },
      {{"prefill_cache", 0, "present_key"}, {"cache", 1, "present_key"}, {"logits", 1, "logits"}}};
  auto execution = Execution(
      options,
      {{{"first_token", Value<int64_t>({1}, {0})},
        {"next_token", Value<int64_t>({1}, {1})},
        {"empty_cache", Value<float>({1, 1, 0, 1}, {})}}});
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_EQ(execution->State().at("kv").Get<Tensor>().Data<float>(),
            result.component_outputs.at("present_key").Get<Tensor>().Data<float>());
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_TRUE(result.completed);
  const auto& cache = result.outputs.at("cache").Get<Tensor>();
  EXPECT_EQ(cache.Shape(), (TensorShape{1, 1, 2, 1}));
  EXPECT_EQ(cache.Data<float>()[0], 0.f);
  EXPECT_EQ(cache.Data<float>()[1], 1.f);
  EXPECT_EQ(result.outputs.at("prefill_cache").Get<Tensor>().Shape(), (TensorShape{1, 1, 1, 1}));
  const auto* logits = result.outputs.at("logits").Get<Tensor>().Data<float>();
  EXPECT_EQ(logits[0], 0.f);
  EXPECT_EQ(logits[1], 1.f);
  EXPECT_EQ(logits[2], 5.f);
  EXPECT_EQ(logits[3], -2.f);
  execution->Reset();
  EXPECT_EQ(cache.Data<float>()[1], 1.f);
}

TEST(SessionChainDefinitionTest, UnloadedAndStringModelsAreRejected) {
  SessionOptions options;
  InferenceSession unloaded(options, GetEnvironment());
  SessionChainOptions config{{{"bad", &unloaded, {{"input", External("value")}}, {}}}, {{"result", 0, "output"}}};
  std::shared_ptr<const SessionChain> chain;
  EXPECT_FALSE(SessionChain::Create(config, chain).IsOK());
  auto model = IdentityModel("strings", ONNX_NAMESPACE::TensorProto_DataType_STRING, {});
  const auto bytes = model.SerializeAsString();
  ASSERT_STATUS_OK(unloaded.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(unloaded.Initialize());
  EXPECT_FALSE(SessionChain::Create(config, chain).IsOK());
}

class CacheTestAllocator : public IAllocator {
 public:
  explicit CacheTestAllocator(bool gpu = false)
      : IAllocator(OrtMemoryInfo("CacheTest", OrtDeviceAllocator,
                                 gpu ? OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                                 OrtDevice::VendorIds::NVIDIA, 0)
                                     : OrtDevice{})),
        inner_(TestCPUExecutionProvider()->CreatePreferredAllocators()[0]) {}
  void* Alloc(size_t size) override {
    ++attempts;
    if (fail) {
      return nullptr;
    }
    if (throw_on_alloc) {
      ORT_THROW("injected cache allocation failure");
    }
    auto* value = inner_->Alloc(size);
    if (value) {
      ++outstanding;
    }
    return value;
  }
  void Free(void* value) override {
    if (value) {
      --outstanding;
      inner_->Free(value);
    }
  }
  size_t attempts = 0;
  size_t outstanding = 0;
  bool fail = false;
  bool throw_on_alloc = false;

 private:
  AllocatorPtr inner_;
};

class KVCacheManagerTest : public testing::Test {
 protected:
  KVCacheSpec Spec(std::string name = "key", TensorShapeVector shape = {1, 1, 0, 1},
                   int64_t axis = 2, int64_t capacity = 0) {
    return {std::move(name), ONNX_NAMESPACE::TensorProto_DataType_FLOAT, std::move(shape), axis, capacity};
  }
  template <typename T>
  OrtValue Value(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
    OrtValue value;
    CreateMLValue<T>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], shape, data, &value);
    return value;
  }
  std::unique_ptr<KVCacheManager> Manager(InlinedVector<KVCacheSpec> specs) {
    std::unique_ptr<KVCacheManager> manager;
    ORT_THROW_IF_ERROR(KVCacheManager::Create(std::move(specs), allocator_, manager));
    return manager;
  }
  KVCacheView View(const KVCacheManager& manager, const char* name = "key") {
    KVCacheView view;
    ORT_THROW_IF_ERROR(manager.Get(name, view));
    return view;
  }
  std::shared_ptr<CacheTestAllocator> allocator_ = std::make_shared<CacheTestAllocator>();
};

TEST_F(KVCacheManagerTest, EmptyReserveGrowthResetAndOwnership) {
  auto manager = Manager({Spec("key", {1, 1, 0, 1}, 2, 3)});
  EXPECT_EQ(View(*manager).length, 0);
  EXPECT_EQ(View(*manager).capacity, 3);
  EXPECT_EQ(manager->Stats().allocations, 1u);
  ASSERT_STATUS_OK(manager->Reserve("key", 2));
  EXPECT_EQ(manager->Stats().allocations, 1u);
  ASSERT_STATUS_OK(manager->Reserve("key", 4));
  EXPECT_EQ(View(*manager).capacity, 4);
  auto values = Value<float>({1, 1, 2, 1}, {1.f, 2.f});
  ASSERT_STATUS_OK(manager->Update("key", values, KVCacheUpdateKind::Copy));
  auto retained = View(*manager);
  EXPECT_EQ(retained.length, 2);
  EXPECT_EQ(retained.capacity, 4);
  EXPECT_NE(retained.value.Get<Tensor>().DataRaw(), values.Get<Tensor>().DataRaw());
  ASSERT_STATUS_OK(manager->Reserve("key", 6));
  EXPECT_EQ(View(*manager).capacity, 6);
  EXPECT_EQ(View(*manager).value.Get<Tensor>().Data<float>()[1], 2.f);
  manager->Reset();
  EXPECT_EQ(View(*manager).length, 0);
  EXPECT_EQ(View(*manager).value.Get<Tensor>().Shape()[2], 0);
  EXPECT_EQ(View(*manager).capacity, 6);
  ASSERT_STATUS_OK(manager->Update("key", Value<float>({1, 1, 1, 1}, {9.f}), KVCacheUpdateKind::Append));
  EXPECT_EQ(View(*manager).value.Get<Tensor>().Data<float>()[0], 9.f);
  EXPECT_EQ(retained.value.Get<Tensor>().Data<float>()[0], 1.f);
  EXPECT_EQ(retained.value.Get<Tensor>().Data<float>()[1], 2.f);
  manager.reset();
  values = {};
  EXPECT_EQ(allocator_->outstanding, 1u);
  EXPECT_EQ(retained.value.Get<Tensor>().Data<float>()[1], 2.f);
  retained = {};
  EXPECT_EQ(allocator_->outstanding, 0u);
}

TEST_F(KVCacheManagerTest, SharesOwnedOutputsAndCopiesBorrowedBuffers) {
  auto manager = Manager({Spec()});
  auto owned = Value<float>({1, 1, 2, 1}, {3.f, 4.f});
  ASSERT_STATUS_OK(manager->Update("key", owned, KVCacheUpdateKind::Share));
  auto shared = View(*manager);
  EXPECT_EQ(shared.value.Get<Tensor>().DataRaw(), owned.Get<Tensor>().DataRaw());
  EXPECT_EQ(shared.capacity, 2);
  EXPECT_EQ(manager->Stats().shared_updates, 1u);
  EXPECT_EQ(manager->Stats().bytes_copied, 0u);
  EXPECT_EQ(manager->Stats().allocations, 0u);
  owned = {};
  manager.reset();
  EXPECT_EQ(shared.value.Get<Tensor>().Data<float>()[1], 4.f);

  manager = Manager({Spec()});
  std::array<float, 2> raw{5.f, 6.f};
  OrtValue borrowed;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{1, 1, 2, 1},
                       raw.data(), allocator_->Info(), borrowed);
  EXPECT_FALSE(manager->Update("key", borrowed, KVCacheUpdateKind::Share).IsOK());
  ASSERT_STATUS_OK(manager->Update("key", borrowed, KVCacheUpdateKind::Copy));
  raw[0] = -1.f;
  EXPECT_EQ(View(*manager).value.Get<Tensor>().Data<float>()[0], 5.f);
  EXPECT_EQ(manager->Stats().bytes_copied, 2 * sizeof(float));
  EXPECT_EQ(manager->Stats().allocations, 1u);
}

TEST_F(KVCacheManagerTest, AppendsAlongArbitraryAxesWithCompactPacking) {
  auto manager = Manager({Spec("key", {2, 0, 2}, 1, 3)});
  auto first = Value<float>({2, 1, 2}, {1.f, 2.f, 10.f, 20.f});
  auto second = Value<float>({2, 2, 2}, {3.f, 4.f, 5.f, 6.f, 30.f, 40.f, 50.f, 60.f});
  ASSERT_STATUS_OK(manager->Update("key", first, KVCacheUpdateKind::Append));
  auto old = View(*manager);
  ASSERT_STATUS_OK(manager->Update("key", second, KVCacheUpdateKind::Append));
  auto current = View(*manager);
  EXPECT_EQ(current.length, 3);
  EXPECT_EQ(current.capacity, 3);
  constexpr std::array<float, 12> expected{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(current.value.Get<Tensor>().Data<float>()[i], expected[i]);
  }
  EXPECT_EQ(old.value.Get<Tensor>().Data<float>()[2], 10.f);
  const auto allocations = manager->Stats().allocations;
  ASSERT_STATUS_OK(manager->Update("key", Value<float>({2, 0, 2}, {}), KVCacheUpdateKind::Append));
  EXPECT_EQ(manager->Stats().allocations, allocations);
  EXPECT_EQ(View(*manager).value.Get<Tensor>().DataRaw(), current.value.Get<Tensor>().DataRaw());
  ASSERT_STATUS_OK(manager->Reserve("key", 7));
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(View(*manager).value.Get<Tensor>().Data<float>()[i], expected[i]);
  }
  manager = Manager({Spec("key", {0, 2}, 0)});
  ASSERT_STATUS_OK(manager->Update("key", Value<float>({1, 2}, {1.f, 2.f}), KVCacheUpdateKind::Append));
  ASSERT_STATUS_OK(manager->Update("key", Value<float>({1, 2}, {3.f, 4.f}), KVCacheUpdateKind::Append));
  EXPECT_EQ(View(*manager).length, 2);
  EXPECT_EQ(View(*manager).value.Get<Tensor>().Data<float>()[3], 4.f);
  manager = Manager({Spec("key", {2, 2, 0}, 2)});
  ASSERT_STATUS_OK(manager->Update("key", Value<float>({2, 2, 1}, {1.f, 2.f, 3.f, 4.f}),
                                   KVCacheUpdateKind::Append));
  ASSERT_STATUS_OK(manager->Update("key", Value<float>({2, 2, 1}, {10.f, 20.f, 30.f, 40.f}),
                                   KVCacheUpdateKind::Append));
  constexpr std::array<float, 8> last_axis{1.f, 10.f, 2.f, 20.f, 3.f, 30.f, 4.f, 40.f};
  for (size_t i = 0; i < last_axis.size(); ++i) {
    EXPECT_EQ(View(*manager).value.Get<Tensor>().Data<float>()[i], last_axis[i]);
  }
}

TEST_F(KVCacheManagerTest, MultipleLayersAndFloatingPointTypes) {
  auto half = Spec("layer0.key");
  half.element_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  auto bfloat = Spec("layer0.value");
  bfloat.element_type = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  auto manager = Manager({half, bfloat, Spec("layer1.key"), Spec("layer1.value")});
  ASSERT_STATUS_OK(manager->Update("layer0.key", Value<MLFloat16>({1, 1, 1, 1}, {MLFloat16(2.f)}),
                                   KVCacheUpdateKind::Copy));
  ASSERT_STATUS_OK(manager->Update("layer0.value", Value<BFloat16>({1, 1, 1, 1}, {BFloat16(3.f)}),
                                   KVCacheUpdateKind::Append));
  for (const char* name : {"layer1.key", "layer1.value"}) {
    ASSERT_STATUS_OK(manager->Update(name, Value<float>({1, 1, 1, 1}, {4.f}), KVCacheUpdateKind::Share));
    EXPECT_EQ(View(*manager, name).length, 1);
  }
  EXPECT_EQ(View(*manager, "layer0.key").value.Get<Tensor>().Data<MLFloat16>()[0].ToFloat(), 2.f);
  EXPECT_EQ(View(*manager, "layer0.value").value.Get<Tensor>().Data<BFloat16>()[0].ToFloat(), 3.f);
  manager->Reset();
  EXPECT_EQ(View(*manager, "layer0.value").length, 0);
  ASSERT_STATUS_OK(manager->Update("layer1.key", Value<float>({1, 1, 0, 1}, {}), KVCacheUpdateKind::Copy));
}

TEST_F(KVCacheManagerTest, RejectsInvalidLayoutsDevicesAndUpdates) {
  std::unique_ptr<KVCacheManager> manager;
  EXPECT_FALSE(KVCacheManager::Create({}, allocator_, manager).IsOK());
  EXPECT_FALSE(KVCacheManager::Create({Spec()}, nullptr, manager).IsOK());
  EXPECT_FALSE(KVCacheManager::Create({Spec()}, std::make_shared<CacheTestAllocator>(true), manager).IsOK());
  auto reject = [&](KVCacheSpec spec) {
    EXPECT_FALSE(KVCacheManager::Create({std::move(spec)}, allocator_, manager).IsOK());
  };
  auto spec = Spec();
  spec.name.clear();
  reject(spec);
  EXPECT_FALSE(KVCacheManager::Create({Spec(), Spec()}, allocator_, manager).IsOK());
  for (int type : std::array<int, 3>{
           ONNX_NAMESPACE::TensorProto_DataType_INT64, ONNX_NAMESPACE::TensorProto_DataType_STRING, 999}) {
    spec = Spec();
    spec.element_type = type;
    reject(spec);
  }
  for (int64_t axis : {-1, 4}) {
    spec = Spec();
    spec.sequence_axis = axis;
    reject(spec);
  }
  spec = Spec();
  spec.initial_capacity = -1;
  reject(spec);
  for (auto shape : {TensorShapeVector{}, TensorShapeVector{1, 0, 0, 1}, TensorShapeVector{1, 1, 2, 1},
                     TensorShapeVector{1, -1, 0, 1}}) {
    spec = Spec();
    spec.shape = std::move(shape);
    reject(spec);
  }
  manager = Manager({Spec()});
  KVCacheView untouched;
  untouched.length = 77;
  EXPECT_FALSE(manager->Get("missing", untouched).IsOK());
  EXPECT_EQ(untouched.length, 77);
  EXPECT_FALSE(manager->Reserve("missing", 2).IsOK());
  EXPECT_FALSE(manager->Reserve("key", -1).IsOK());
  EXPECT_FALSE(manager->Update("missing", {}, KVCacheUpdateKind::Copy).IsOK());
  EXPECT_FALSE(manager->Update("key", {}, KVCacheUpdateKind::Copy).IsOK());
  EXPECT_FALSE(manager->Update("key", Value<int64_t>({1, 1, 1, 1}, {1}), KVCacheUpdateKind::Copy).IsOK());
  EXPECT_FALSE(manager->Update("key", Value<float>({1}, {1.f}), KVCacheUpdateKind::Copy).IsOK());
  EXPECT_FALSE(manager->Update("key", Value<float>({1, 2, 1, 1}, {1.f, 2.f}), KVCacheUpdateKind::Copy).IsOK());
  auto valid = Value<float>({1, 1, 2, 1}, {1.f, 2.f});
  EXPECT_FALSE(manager->Update("key", valid, static_cast<KVCacheUpdateKind>(99)).IsOK());
  ASSERT_STATUS_OK(manager->Update("key", valid, KVCacheUpdateKind::Copy));
  for (auto kind : {KVCacheUpdateKind::Copy, KVCacheUpdateKind::Share}) {
    EXPECT_FALSE(manager->Update("key", Value<float>({1, 1, 1, 1}, {0.f}), kind).IsOK());
  }
  float raw = 0.f;
  OrtValue device;
  OrtMemoryInfo gpu("Cuda", OrtDeviceAllocator,
                    OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0));
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{1, 1, 1, 1}, &raw, gpu, device);
  EXPECT_FALSE(manager->Update("key", device, KVCacheUpdateKind::Copy).IsOK());
#ifdef ENABLE_STRIDED_TENSORS
  std::array<float, 3> storage{};
  OrtValue strided;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{1, 1, 2, 1},
                       storage.data(), allocator_->Info(), strided, 0, std::array<int64_t, 4>{4, 4, 2, 1});
  EXPECT_FALSE(manager->Update("key", strided, KVCacheUpdateKind::Copy).IsOK());
#endif
}

TEST_F(KVCacheManagerTest, CheckedSizesAndAllocationFailurePreserveState) {
  std::unique_ptr<KVCacheManager> manager;
  const auto maximum = std::numeric_limits<int64_t>::max();
  EXPECT_ANY_THROW(ORT_THROW_IF_ERROR(KVCacheManager::Create({Spec("key", {maximum, 0, maximum}, 1, 1)},
                                                             allocator_, manager)));
  EXPECT_ANY_THROW(ORT_THROW_IF_ERROR(KVCacheManager::Create({Spec("key", {maximum, 0}, 1, 1)},
                                                             allocator_, manager)));
  EXPECT_EQ(allocator_->attempts, 0u);
  allocator_->fail = true;
  EXPECT_FALSE(KVCacheManager::Create({Spec("key", {1, 1, 0, 1}, 2, 1)}, allocator_, manager).IsOK());
  EXPECT_EQ(manager, nullptr);
  allocator_->fail = false;
  manager = Manager({Spec()});
  ASSERT_STATUS_OK(manager->Update("key", Value<float>({1, 1, 1, 1}, {5.f}), KVCacheUpdateKind::Copy));
  auto old = View(*manager);
  allocator_->fail = true;
  EXPECT_FALSE(manager->Reserve("key", 10).IsOK());
  EXPECT_FALSE(manager->Update("key", Value<float>({1, 1, 2, 1}, {6.f, 7.f}), KVCacheUpdateKind::Copy).IsOK());
  EXPECT_FALSE(manager->Update("key", Value<float>({1, 1, 1, 1}, {6.f}), KVCacheUpdateKind::Append).IsOK());
  allocator_->fail = false;
  allocator_->throw_on_alloc = true;
  EXPECT_THROW(ORT_THROW_IF_ERROR(manager->Reserve("key", 10)), OnnxRuntimeException);
  allocator_->throw_on_alloc = false;
  EXPECT_EQ(View(*manager).length, 1);
  EXPECT_EQ(View(*manager).value.Get<Tensor>().DataRaw(), old.value.Get<Tensor>().DataRaw());
  EXPECT_EQ(View(*manager).value.Get<Tensor>().Data<float>()[0], 5.f);

  auto half = Spec("key", {0}, 0);
  half.element_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  manager = Manager({half});
  ASSERT_STATUS_OK(manager->Update("key", Value<MLFloat16>({1}, {MLFloat16(0.f)}), KVCacheUpdateKind::Copy));
  MLFloat16 raw(0.f);
  OrtValue oversized;
  Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(), TensorShape{maximum}, &raw, allocator_->Info(), oversized);
  const auto attempts = allocator_->attempts;
  EXPECT_ANY_THROW(ORT_THROW_IF_ERROR(manager->Update("key", oversized, KVCacheUpdateKind::Append)));
  EXPECT_EQ(allocator_->attempts, attempts);
  EXPECT_EQ(View(*manager).length, 1);
}

LoopKVCacheBinding KeyBinding() {
  return {{"key", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, 0, 1}, 2, 0},
          "past_key",
          "present_key",
          KVCacheUpdateKind::Share};
}

TEST_P(SessionChainTest, ManagedCpuCacheInLoopRetainsAndResets) {
  auto session = Session(stateful::MakeDecoderModel());
  SessionLoopOptions options;
  options.finite_input_names = {"token"};
  options.output_names = {"logits"};
  options.exit_conditions = {{LoopExitKind::EndOfInputs}};
  options.kv_caches = {KeyBinding()};
  auto loop = Loop(*session, options);
  SessionLoopInputs inputs{{}, {{{"token", Value<int64_t>({1}, {0})}}, {{"token", Value<int64_t>({1}, {1})}}}};
  std::unique_ptr<SessionLoopExecution> execution;
  ASSERT_STATUS_OK(loop->CreateExecution(inputs, execution));
  ASSERT_NE(execution->Cache(), nullptr);
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  KVCacheView first;
  ASSERT_STATUS_OK(execution->Cache()->Get("key", first));
  EXPECT_EQ(first.length, 1);
  ASSERT_STATUS_OK(execution->Step(result));
  KVCacheView last;
  ASSERT_STATUS_OK(execution->Cache()->Get("key", last));
  EXPECT_EQ(last.length, 2);
  EXPECT_EQ(last.value.Get<Tensor>().Data<float>()[1], 1.f);
  EXPECT_EQ(first.value.Get<Tensor>().Shape()[2], 1);
  EXPECT_EQ(execution->Cache()->Stats().shared_updates, 2u);
  execution->Reset();
  ASSERT_STATUS_OK(execution->Cache()->Get("key", last));
  EXPECT_EQ(last.length, 0);
  ASSERT_STATUS_OK(execution->Run(result));
  ASSERT_STATUS_OK(execution->Cache()->Get("key", last));
  EXPECT_EQ(last.length, 2);
  EXPECT_EQ(first.value.Get<Tensor>().Data<float>()[0], 0.f);
}

TEST_P(SessionChainTest, ManagedCacheSharesAcrossPrefillDecodeAndIndependentContexts) {
  auto prefill = Session(stateful::MakeDecoderModel());
  auto decode = Session(stateful::MakeDecoderModel());
  SessionChainComponent first{
      "prefill",
      prefill.get(),
      {{"token", External("prompt")}},
      {},
      {KeyBinding()},
  };
  first.kv_caches[0].cache.initial_capacity = 4;
  SessionLoopOptions options;
  options.finite_input_names = {"token"};
  options.output_names = {"logits", "present_key"};
  options.exit_conditions = {{LoopExitKind::EndOfInputs}};
  options.kv_caches = {KeyBinding()};
  options.kv_caches[0].cache.initial_capacity = 4;
  auto loop = Loop(*decode, options);
  SessionChainComponent second{
      "decode",
      loop,
      {{"token", Sequence("tokens")}},
      {},
      {},
  };
  SessionChainOptions chain_options{{first, second}, {{"cache", 1, "present_key"}, {"logits", 1, "logits"}}};
  std::shared_ptr<const SessionChain> chain;
  ASSERT_STATUS_OK(SessionChain::Create(chain_options, chain));
  SessionChainInputs inputs{{{"prompt", Value<int64_t>({1}, {0})}},
                            {{"tokens", {Value<int64_t>({1}, {1}), Value<int64_t>({1}, {2})}}}};
  std::unique_ptr<SessionChainExecution> execution, independent;
  ASSERT_STATUS_OK(chain->CreateExecution(inputs, execution));
  ASSERT_STATUS_OK(chain->CreateExecution(inputs, independent));
  ASSERT_NE(execution->Cache(), nullptr);
  EXPECT_NE(execution->Cache(), independent->Cache());
  KVCacheView empty;
  ASSERT_STATUS_OK(execution->Cache()->Get("key", empty));
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  auto prefill_snapshot = result.component_outputs.at("present_key");
  EXPECT_NE(prefill_snapshot.Get<Tensor>().DataRaw(), empty.value.Get<Tensor>().DataRaw());
  EXPECT_EQ(execution->State().at("key").Get<Tensor>().DataRaw(), prefill_snapshot.Get<Tensor>().DataRaw());
  EXPECT_EQ(independent->State().at("key").Get<Tensor>().Shape()[2], 0);
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.outputs.at("cache").Get<Tensor>().Shape()[2], 3);
  EXPECT_EQ(result.outputs.at("cache").Get<Tensor>().Data<float>()[2], 2.f);
  EXPECT_EQ(execution->Cache()->Stats().shared_updates, 3u);
  EXPECT_EQ(execution->Cache()->Stats().bytes_copied, 0u);
  EXPECT_EQ(prefill_snapshot.Get<Tensor>().Shape()[2], 1);
  auto retained = result.outputs.at("cache");
  execution->Reset();
  EXPECT_EQ(execution->State().at("key").Get<Tensor>().Shape()[2], 0);
  ASSERT_STATUS_OK(execution->Run(result));
  execution.reset();
  independent.reset();
  chain.reset();
  chain_options = {};
  first = {};
  second = {};
  loop.reset();
  prefill.reset();
  decode.reset();
  EXPECT_EQ(retained.Get<Tensor>().Data<float>()[2], 2.f);
}

TEST_P(SessionChainTest, OrdinaryStateBindingsCanReadAndCommitManagedCache) {
  auto first = Session(stateful::MakeDecoderModel());
  auto second = Session(stateful::MakeDecoderModel());
  SessionChainOptions options{
      {
          {"first", first.get(), {{"token", External("token")}}, {}, {KeyBinding()}},
          {
              "second",
              second.get(),
              {{"token", External("token")}, {"past_key", State("key")}},
              {{"key", "present_key"}},
              {},
          },
      },
      {{"cache", 1, "present_key"}}};
  auto execution = Execution(options, {{{"token", Value<int64_t>({1}, {1})}}});
  SessionChainStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  KVCacheView view;
  ASSERT_STATUS_OK(execution->Cache()->Get("key", view));
  EXPECT_EQ(view.length, 2);
  EXPECT_EQ(view.value.Get<Tensor>().DataRaw(), execution->State().at("key").Get<Tensor>().DataRaw());
}

TEST_P(SessionChainTest, ManagedCacheDefinitionsRejectConflictsAndIncompatibleMetadata) {
  auto session = Session(stateful::MakeDecoderModel());
  SessionLoopOptions options;
  options.output_names = {"logits"};
  options.exit_conditions = {{LoopExitKind::MaxIterations, 1}};
  options.kv_caches = {KeyBinding()};
  std::shared_ptr<const SessionLoop> loop;
  auto reject_loop = [&](SessionLoopOptions config) {
    EXPECT_FALSE(SessionLoop::Create(*session, std::move(config), loop).IsOK());
    EXPECT_EQ(loop, nullptr);
  };
  auto invalid = options;
  invalid.kv_caches[0].input_name = "missing";
  reject_loop(invalid);
  invalid = options;
  invalid.kv_caches[0].output_name = "missing";
  reject_loop(invalid);
  invalid = options;
  invalid.kv_caches[0].update_kind = static_cast<KVCacheUpdateKind>(99);
  reject_loop(invalid);
  invalid = options;
  invalid.kv_caches[0].cache.element_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  reject_loop(invalid);
  invalid = options;
  invalid.kv_caches[0].cache.shape = {1, 0, 1};
  invalid.kv_caches[0].cache.sequence_axis = 1;
  reject_loop(invalid);
  invalid = options;
  invalid.kv_caches[0].cache.shape = {2, 1, 0, 1};
  reject_loop(invalid);
  invalid = options;
  invalid.finite_input_names = {"past_key"};
  reject_loop(invalid);
  ASSERT_STATUS_OK(SessionLoop::Create(*session, options, loop));
  std::unique_ptr<SessionLoopExecution> execution;
  EXPECT_FALSE(loop->CreateExecution(
                       {{{"past_key", Value<float>({1, 1, 0, 1}, {})}}, {}}, execution)
                   .IsOK());

  SessionChainOptions chain_options{
      {
          {"first", session.get(), {{"token", External("token")}}, {}, {KeyBinding()}},
          {"second", session.get(), {{"token", External("token")}}, {}, {KeyBinding()}},
      },
      {{"cache", 1, "present_key"}}};
  std::shared_ptr<const SessionChain> chain;
  chain_options.components[1].kv_caches[0].cache.initial_capacity = 2;
  EXPECT_FALSE(SessionChain::Create(chain_options, chain).IsOK());
  chain_options.components[1].kv_caches[0].cache.initial_capacity = 0;
  chain_options.components[0].state_outputs = {{"key", "present_key"}};
  EXPECT_FALSE(SessionChain::Create(chain_options, chain).IsOK());
  chain_options.components[0].state_outputs.clear();
  chain_options.components[0].inputs = {{"past_key", External("token")}};
  EXPECT_FALSE(SessionChain::Create(chain_options, chain).IsOK());
}

TEST_P(SessionChainTest, ManagedCacheCopyAndDeltaPolicies) {
  auto full = Session(stateful::MakeDecoderModel());
  SessionLoopOptions options;
  options.finite_input_names = {"token"};
  options.output_names = {"logits"};
  options.exit_conditions = {{LoopExitKind::EndOfInputs}};
  options.kv_caches = {KeyBinding()};
  options.kv_caches[0].update_kind = KVCacheUpdateKind::Copy;
  auto loop = Loop(*full, options);
  SessionLoopInputs inputs{{}, {{{"token", Value<int64_t>({1}, {0})}}, {{"token", Value<int64_t>({1}, {1})}}}};
  std::unique_ptr<SessionLoopExecution> execution;
  ASSERT_STATUS_OK(loop->CreateExecution(inputs, execution));
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(execution->Cache()->Stats().bytes_copied, 3 * sizeof(float));
  EXPECT_EQ(execution->Cache()->Stats().shared_updates, 0u);

  auto model = stateful::MakeDecoderModel();
  auto& graph = *model.mutable_graph();
  graph.mutable_node()->RemoveLast();
  graph.mutable_output(1)->set_name("new_key");
  graph.mutable_output(1)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_value(1);
  stateful::SetTensorType(*graph.add_output(), "observed_past",
                          ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 1, -1, 1});
  stateful::AddNode(graph, "Identity", {"past_key"}, "observed_past");
  auto delta = Session(model);
  options.output_names = {"new_key", "observed_past"};
  options.kv_caches[0].output_name = "new_key";
  options.kv_caches[0].update_kind = KVCacheUpdateKind::Append;
  loop = Loop(*delta, options);
  execution.reset();
  ASSERT_STATUS_OK(loop->CreateExecution(inputs, execution));
  ASSERT_STATUS_OK(execution->Run(result));
  KVCacheView view;
  ASSERT_STATUS_OK(execution->Cache()->Get("key", view));
  EXPECT_EQ(view.length, 2);
  EXPECT_EQ(view.value.Get<Tensor>().Data<float>()[0], 0.f);
  EXPECT_EQ(view.value.Get<Tensor>().Data<float>()[1], 1.f);
  EXPECT_EQ(result.outputs[1].Get<Tensor>().Shape()[2], 1);
  EXPECT_EQ(result.outputs[1].Get<Tensor>().Data<float>()[0], 0.f);
}

TEST_P(SessionChainTest, InvalidManagedOutputRequiresResetWithoutCommittingCache) {
  auto session = Session(stateful::MakeDecoderModel());
  SessionLoopOptions options;
  options.output_names = {"present_key"};
  options.exit_conditions = {{LoopExitKind::MaxIterations, 3}};
  options.kv_caches = {KeyBinding()};
  auto loop = Loop(*session, options);
  std::unique_ptr<SessionLoopExecution> execution;
  ASSERT_STATUS_OK(loop->CreateExecution({{{"token", Value<int64_t>({1}, {0})}}, {}}, execution));
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  session->after_run = [&](IOBinding& binding) {
    binding.GetOutputs()[0] = Value<float>({1, 2, 1, 1}, {9.f, 9.f});
    return Status::OK();
  };
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(result.iterations, 1);
  KVCacheView view;
  ASSERT_STATUS_OK(execution->Cache()->Get("key", view));
  EXPECT_EQ(view.length, 1);
  EXPECT_EQ(view.value.Get<Tensor>().Data<float>()[0], 0.f);
  session->after_run = {};
  execution->Reset();
  ASSERT_STATUS_OK(execution->Run(result));
  ASSERT_STATUS_OK(execution->Cache()->Get("key", view));
  EXPECT_EQ(view.length, 3);
}

INSTANTIATE_TEST_SUITE_P(Cpu, SessionChainTest, testing::Values(false));
#ifdef USE_CUDA
INSTANTIATE_TEST_SUITE_P(Cuda, SessionChainTest, testing::Values(true));
#endif

}  // namespace
}  // namespace onnxruntime::test
