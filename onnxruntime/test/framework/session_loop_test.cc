// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <future>
#include <functional>
#include <memory>

#include "gtest/gtest.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_viewer.h"
#include "core/session/inference_session.h"
#include "core/session/IOBinding.h"
#include "core/session/session_loop.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/stateful_execution_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

class ControlledSession : public InferenceSession {
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

class SessionLoopTest : public testing::TestWithParam<bool> {
 protected:
  std::unique_ptr<ControlledSession> Session(ONNX_NAMESPACE::ModelProto model) {
    SessionOptions options;
    options.intra_op_param.thread_pool_size = 1;
    options.inter_op_param.thread_pool_size = 1;
    options.graph_optimization_level = TransformerLevel::Default;
    auto session = std::make_unique<ControlledSession>(options, GetEnvironment());
    if (GetParam()) {
      auto provider = DefaultCudaExecutionProvider();
      EXPECT_NE(provider, nullptr);
      ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(std::move(provider)));
    }
    const auto bytes = model.SerializeAsString();
    ORT_THROW_IF_ERROR(session->Load(bytes.data(), static_cast<int>(bytes.size())));
    ORT_THROW_IF_ERROR(session->Initialize());
    if (GetParam()) {
      const auto& graph = session->GetSessionState().GetGraphViewer();
      for (const auto& node : graph.Nodes()) {
        EXPECT_EQ(node.GetExecutionProviderType(), kCudaExecutionProvider);
      }
    }
    return session;
  }

  template <typename T>
  OrtValue Value(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
    OrtValue value;
    CreateMLValue<T>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], shape, data, &value);
    return value;
  }

  OrtValue Scalar(float value) { return Value<float>({}, {value}); }

  SessionLoopOptions Options() {
    SessionLoopOptions options;
    options.finite_input_names = {"input_value"};
    options.feedback = {{"past_value", "present_value"}};
    options.output_names = {"present_value"};
    options.exit_conditions = {{LoopExitKind::EndOfInputs}};
    return options;
  }

  SessionLoopInputs Inputs() {
    return {{{"past_value", Scalar(0.f)}},
            {{{"input_value", Scalar(2.f)}}, {{"input_value", Scalar(3.f)}}, {{"input_value", Scalar(4.f)}}}};
  }

  std::unique_ptr<SessionLoopExecution> Execution(
      InferenceSession& session, SessionLoopOptions options, SessionLoopInputs inputs) {
    std::shared_ptr<const SessionLoop> loop;
    ORT_THROW_IF_ERROR(SessionLoop::Create(session, std::move(options), loop));
    std::unique_ptr<SessionLoopExecution> execution;
    ORT_THROW_IF_ERROR(loop->CreateExecution(std::move(inputs), execution));
    return execution;
  }
};

TEST_P(SessionLoopTest, RetentionResetCompletionAndRunParity) {
  auto session = Session(stateful::MakeFeedbackModel());
  std::shared_ptr<const SessionLoop> loop;
  ASSERT_STATUS_OK(SessionLoop::Create(*session, Options(), loop));
  EXPECT_EQ(loop->Options().finite_input_names.size(), 1u);
  std::unique_ptr<SessionLoopExecution> first, second;
  ASSERT_STATUS_OK(loop->CreateExecution(Inputs(), first));
  ASSERT_STATUS_OK(loop->CreateExecution(Inputs(), second));
  loop.reset();
  SessionLoopStepResult result;
  InlinedVector<SessionLoopStepResult> retained;
  for (float expected : {2.f, 5.f, 9.f}) {
    ASSERT_STATUS_OK(first->Step(result));
    EXPECT_TRUE(result.executed);
    EXPECT_EQ(*result.outputs[0].Get<Tensor>().Data<float>(), expected);
    retained.push_back(result);
  }
  EXPECT_TRUE(result.completed);
  EXPECT_EQ(result.iterations, 3);
  EXPECT_EQ(result.generated_tokens, 0);
  EXPECT_EQ(result.exit_reasons, (InlinedVector<LoopExitKind>{LoopExitKind::EndOfInputs}));
  ASSERT_STATUS_OK(second->Run(result));
  EXPECT_EQ(*result.outputs[0].Get<Tensor>().Data<float>(), 9.f);
  const auto calls = session->calls;
  ASSERT_STATUS_OK(first->Step(result));
  EXPECT_FALSE(result.executed);
  EXPECT_TRUE(result.outputs.empty());
  EXPECT_EQ(session->calls, calls);
  ASSERT_STATUS_OK(first->Run(result));
  EXPECT_FALSE(result.executed);
  first->Reset();
  ASSERT_STATUS_OK(first->Step(result));
  EXPECT_EQ(result.iterations, 1);
  EXPECT_EQ(*result.outputs[0].Get<Tensor>().Data<float>(), 2.f);
  EXPECT_EQ(*retained[0].outputs[0].Get<Tensor>().Data<float>(), 2.f);
  EXPECT_EQ(*retained[1].outputs[0].Get<Tensor>().Data<float>(), 5.f);
  EXPECT_EQ(*retained[2].outputs[0].Get<Tensor>().Data<float>(), 9.f);
}

TEST_P(SessionLoopTest, ExplicitExitsAndZeroIteration) {
  auto session = Session(stateful::MakeFeedbackModel());
  auto options = Options();
  options.exit_conditions = {
      {LoopExitKind::BooleanOutput, 0, "done"},
      {LoopExitKind::MaxIterations, 3},
      {LoopExitKind::EndOfInputs}};
  auto execution = Execution(*session, options, Inputs());
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.iterations, 3);
  EXPECT_EQ(result.exit_reasons,
            (InlinedVector<LoopExitKind>{LoopExitKind::EndOfInputs, LoopExitKind::MaxIterations,
                                         LoopExitKind::BooleanOutput}));
  options.exit_conditions = {{LoopExitKind::MaxIterations, 1}};
  execution = Execution(*session, options, Inputs());
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.iterations, 1);
  options.exit_conditions = {{LoopExitKind::MaxIterations, 0}};
  execution = Execution(*session, options, Inputs());
  const auto calls = session->calls;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.iterations, 0);
  EXPECT_FALSE(result.executed);
  EXPECT_EQ(session->calls, calls);

  execution = Execution(*session, Options(), {{{"past_value", Scalar(0.f)}}, {}});
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_TRUE(result.completed);
  EXPECT_EQ(result.iterations, 0);
}

TEST_P(SessionLoopTest, InvariantsNoFeedbackAndFiniteExhaustion) {
  auto session = Session(stateful::MakeFeedbackModel());
  auto options = Options();
  options.finite_input_names.clear();
  options.feedback.clear();
  options.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
  auto execution = Execution(*session, options, {{{"input_value", Scalar(2.f)}, {"past_value", Scalar(3.f)}}, {}});
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(*result.outputs[0].Get<Tensor>().Data<float>(), 5.f);
  EXPECT_EQ(result.iterations, 2);
  options = Options();
  options.exit_conditions = {{LoopExitKind::MaxIterations, 4}};
  execution = Execution(*session, options, Inputs());
  result.iterations = 99;
  EXPECT_FALSE(execution->Run(result).IsOK());
  EXPECT_EQ(result.iterations, 99);
  EXPECT_FALSE(execution->RequiresReset());
}

TEST_P(SessionLoopTest, BooleanCanStopBeforeInputExhaustion) {
  auto session = Session(stateful::MakeFeedbackModel());
  auto options = Options();
  options.exit_conditions = {{LoopExitKind::BooleanOutput, 0, "done"}};
  auto inputs = Inputs();
  inputs.batches[0]["input_value"] = Scalar(6.f);
  auto execution = Execution(*session, options, inputs);
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.iterations, 1);
  EXPECT_EQ(result.exit_reasons, (InlinedVector<LoopExitKind>{LoopExitKind::BooleanOutput}));
}

ONNX_NAMESPACE::ModelProto TokenModel() {
  auto model = stateful::MakeModel("token_output_reference");
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "token_input", ONNX_NAMESPACE::TensorProto_DataType_INT64, {});
  stateful::SetTensorType(*graph.add_output(), "token_output", ONNX_NAMESPACE::TensorProto_DataType_INT64, {});
  stateful::AddNode(graph, "Identity", {"token_input"}, "token_output");
  return model;
}

TEST_P(SessionLoopTest, EosAndGeneratedTokenLimits) {
  auto session = Session(TokenModel());
  SessionLoopOptions options;
  options.finite_input_names = {"token_input"};
  options.output_names = {"token_output"};
  options.generated_token_output = "token_output";
  options.exit_conditions = {{LoopExitKind::Eos, 0, "", {4, 9}},
                             {LoopExitKind::MaxGeneratedTokens, 2}};
  SessionLoopInputs inputs{
      {},
      {{{"token_input", Value<int64_t>({}, {2})}},
       {{"token_input", Value<int64_t>({}, {4})}},
       {{"token_input", Value<int64_t>({}, {8})}}}};
  auto execution = Execution(*session, options, inputs);
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.generated_tokens, 2);
  EXPECT_EQ(*result.outputs[0].Get<Tensor>().Data<int64_t>(), 4);
  EXPECT_EQ(result.exit_reasons,
            (InlinedVector<LoopExitKind>{LoopExitKind::MaxGeneratedTokens, LoopExitKind::Eos}));
  options.exit_conditions = {{LoopExitKind::MaxGeneratedTokens, 0}};
  execution = Execution(*session, options, inputs);
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_FALSE(result.executed);
  EXPECT_EQ(result.generated_tokens, 0);
  options.exit_conditions = {{LoopExitKind::Eos, 0, "", {2}}};
  execution = Execution(*session, options, inputs);
  ASSERT_STATUS_OK(execution->Step(result));
  EXPECT_TRUE(result.completed);
  EXPECT_EQ(result.generated_tokens, 1);
}

TEST_P(SessionLoopTest, CancellationBeforeAndDuringRun) {
  auto session = Session(stateful::MakeFeedbackModel());
  auto execution = Execution(*session, Options(), Inputs());
  execution->Cancel();
  SessionLoopStepResult result;
  result.iterations = 99;
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_EQ(session->calls, 0u);
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(result.iterations, 99);
  execution->Reset();

  std::promise<void> entered, resume;
  auto resumed = resume.get_future();
  session->before_run = [&]() {
    entered.set_value();
    resumed.wait();
    return Status::OK();
  };
  auto status = std::async(std::launch::async, [&]() { return execution->Step(result); });
  entered.get_future().wait();
  execution->Cancel();
  resume.set_value();
  EXPECT_FALSE(status.get().IsOK());
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(result.iterations, 99);
  EXPECT_FALSE(execution->Step(result).IsOK());
  session->before_run = {};
  execution->Reset();
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.iterations, 3);
}

TEST_P(SessionLoopTest, FailureAndExceptionRequireReset) {
  auto session = Session(stateful::MakeFeedbackModel());
  auto execution = Execution(*session, Options(), Inputs());
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  session->before_run = []() { return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "injected provider failure"); };
  auto status = execution->Step(result);
  EXPECT_EQ(status.Code(), common::EP_FAIL);
  EXPECT_NE(status.ErrorMessage().find("injected provider failure"), std::string::npos);
  EXPECT_EQ(result.iterations, 1);
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_FALSE(execution->Run(result).IsOK());
  execution->Reset();
  session->before_run = []() -> Status { ORT_THROW("injected exception"); };
  EXPECT_THROW(ORT_THROW_IF_ERROR(execution->Step(result)), OnnxRuntimeException);
  EXPECT_TRUE(execution->RequiresReset());
  session->before_run = {};
  execution->Reset();
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(*result.outputs[0].Get<Tensor>().Data<float>(), 9.f);
}

TEST_P(SessionLoopTest, RejectsInvalidDefinitions) {
  auto session = Session(stateful::MakeFeedbackModel());
  std::shared_ptr<const SessionLoop> loop;
  auto reject = [&](SessionLoopOptions options) {
    EXPECT_FALSE(SessionLoop::Create(*session, std::move(options), loop).IsOK());
    EXPECT_EQ(loop, nullptr);
  };
  auto options = Options();
  options.exit_conditions.clear();
  reject(options);
  options = Options();
  options.output_names.clear();
  reject(options);
  for (auto names : {InlinedVector<std::string>{"missing"}, InlinedVector<std::string>{"input_value", "input_value"}}) {
    options = Options();
    options.finite_input_names = std::move(names);
    reject(options);
  }
  options = Options();
  options.nonempty_input = "past_value";
  reject(options);
  options = Options();
  options.feedback = {{"input_value", "present_value"}};
  reject(options);
  options.feedback = {{"missing", "present_value"}};
  reject(options);
  options.feedback = {{"past_value", "missing"}};
  reject(options);
  options.feedback = {{"past_value", "done"}};
  reject(options);
  for (auto names : {InlinedVector<std::string>{"missing"}, InlinedVector<std::string>{"done", "done"}}) {
    options = Options();
    options.output_names = std::move(names);
    reject(options);
  }
  options = Options();
  options.generated_token_output = "missing";
  reject(options);
  options.generated_token_output = "present_value";
  reject(options);
  options = Options();
  options.exit_conditions.push_back({LoopExitKind::EndOfInputs});
  reject(options);
  options = Options();
  options.finite_input_names.clear();
  reject(options);
  for (auto kind : {LoopExitKind::MaxIterations, LoopExitKind::MaxGeneratedTokens}) {
    options = Options();
    options.exit_conditions = {{kind, -1}};
    reject(options);
  }
  for (auto kind : {LoopExitKind::MaxGeneratedTokens, LoopExitKind::Eos}) {
    options = Options();
    options.exit_conditions = {{kind, 1}};
    reject(options);
  }
  for (const char* output : {"missing", "present_value"}) {
    options = Options();
    options.exit_conditions = {{LoopExitKind::BooleanOutput, 0, output}};
    reject(options);
  }
  options = Options();
  options.exit_conditions = {{static_cast<LoopExitKind>(99)}};
  reject(options);
}

TEST_P(SessionLoopTest, RejectsInvalidInputsBeforeExecuting) {
  auto session = Session(stateful::MakeFeedbackModel());
  std::shared_ptr<const SessionLoop> loop;
  ASSERT_STATUS_OK(SessionLoop::Create(*session, Options(), loop));
  auto reject = [&](SessionLoopInputs inputs) {
    std::unique_ptr<SessionLoopExecution> execution;
    EXPECT_FALSE(loop->CreateExecution(std::move(inputs), execution).IsOK());
    EXPECT_EQ(execution, nullptr);
  };
  auto inputs = Inputs();
  inputs.inputs.clear();
  reject(inputs);
  inputs = Inputs();
  inputs.inputs = {{"missing", Scalar(0.f)}};
  reject(inputs);
  inputs.inputs = {{"input_value", Scalar(0.f)}};
  reject(inputs);
  inputs = Inputs();
  inputs.inputs["past_value"] = {};
  reject(inputs);
  inputs.inputs["past_value"] = Value<int64_t>({}, {0});
  reject(inputs);
  inputs.inputs["past_value"] = Value<float>({1}, {0.f});
  reject(inputs);
  inputs = Inputs();
  inputs.batches[0].clear();
  reject(inputs);
  inputs = Inputs();
  inputs.batches[0] = {{"missing", Scalar(0.f)}};
  reject(inputs);
  auto options = Options();
  options.nonempty_input = "input_value";
  ASSERT_STATUS_OK(SessionLoop::Create(*session, options, loop));
  inputs = Inputs();
  inputs.batches.clear();
  reject(inputs);
  options.finite_input_names.clear();
  options.nonempty_input.clear();
  options.exit_conditions = {{LoopExitKind::MaxIterations, 1}};
  ASSERT_STATUS_OK(SessionLoop::Create(*session, options, loop));
  reject({{{"input_value", Scalar(1.f)}, {"past_value", Scalar(0.f)}}, {{{"extra", Scalar(1.f)}}}});
  EXPECT_EQ(session->calls, 0u);
}

TEST_P(SessionLoopTest, DynamicShapesEmptyTensorsAndOutputSnapshots) {
  auto model = stateful::MakeModel("dynamic_identity");
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "input", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {-1});
  stateful::SetTensorType(*graph.add_output(), "output", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {-1});
  stateful::AddNode(graph, "Identity", {"input"}, "output");
  auto session = Session(model);
  SessionLoopOptions options;
  options.finite_input_names = {"input"};
  options.output_names = {"output"};
  options.nonempty_input = "input";
  options.exit_conditions = {{LoopExitKind::EndOfInputs}};
  std::shared_ptr<const SessionLoop> loop;
  ASSERT_STATUS_OK(SessionLoop::Create(*session, options, loop));
  std::unique_ptr<SessionLoopExecution> execution;
  EXPECT_FALSE(loop->CreateExecution({{}, {{{"input", Value<float>({0}, {})}}}}, execution).IsOK());
  auto tensor = Value<float>({2}, {1.f, 2.f});
  ASSERT_STATUS_OK(loop->CreateExecution({{}, {{{"input", tensor}}}}, execution));
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  tensor.GetMutable<Tensor>()->MutableData<float>()[0] = 7.f;
  EXPECT_EQ(result.outputs[0].Get<Tensor>().Data<float>()[0], 1.f);
  options.nonempty_input.clear();
  execution = Execution(*session, options, {{}, {{{"input", Value<float>({0}, {})}}}});
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.outputs[0].Get<Tensor>().Shape().Size(), 0);
}

TEST_P(SessionLoopTest, EosDefinitionAndNegativeTokenValidation) {
  auto session = Session(TokenModel());
  SessionLoopOptions options;
  options.output_names = {"token_output"};
  options.generated_token_output = "token_output";
  options.exit_conditions = {{LoopExitKind::Eos}};
  std::shared_ptr<const SessionLoop> loop;
  EXPECT_FALSE(SessionLoop::Create(*session, options, loop).IsOK());
  options.exit_conditions[0].eos_token_ids = {-1};
  EXPECT_FALSE(SessionLoop::Create(*session, options, loop).IsOK());
  options.exit_conditions[0].eos_token_ids = {1};
  auto execution = Execution(*session, options, {{{"token_input", Value<int64_t>({}, {-1})}}, {}});
  SessionLoopStepResult result;
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
}

TEST_P(SessionLoopTest, MetadataTypesAndFixedDimensions) {
  auto model = stateful::MakeModel("fixed_identity");
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "input", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2});
  stateful::SetTensorType(*graph.add_output(), "output", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2});
  stateful::AddNode(graph, "Identity", {"input"}, "output");
  auto session = Session(model);
  SessionLoopOptions options;
  options.output_names = {"output"};
  options.exit_conditions = {{LoopExitKind::MaxIterations, 1}};
  std::shared_ptr<const SessionLoop> loop;
  ASSERT_STATUS_OK(SessionLoop::Create(*session, options, loop));
  std::unique_ptr<SessionLoopExecution> execution;
  EXPECT_FALSE(loop->CreateExecution({{{"input", Value<float>({1}, {1.f})}}, {}}, execution).IsOK());
  ASSERT_STATUS_OK(loop->CreateExecution({{{"input", Value<float>({2}, {1.f, 2.f})}}, {}}, execution));
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Run(result));
  options.generated_token_output = "output";
  EXPECT_FALSE(SessionLoop::Create(*session, options, loop).IsOK());

  options.generated_token_output.clear();
  options.feedback = {{"input", "output"}};
  ASSERT_STATUS_OK(SessionLoop::Create(*session, options, loop));
  ASSERT_STATUS_OK(loop->CreateExecution({{{"input", Value<float>({2}, {1.f, 2.f})}}, {}}, execution));
  ASSERT_STATUS_OK(execution->Run(result));
}

TEST_P(SessionLoopTest, RejectsIncompatibleFeedbackShapes) {
  auto session = Session(stateful::MakeDenoiserModel());
  SessionLoopOptions options;
  options.output_names = {"noise_prediction"};
  options.exit_conditions = {{LoopExitKind::MaxIterations, 1}};
  options.feedback = {{"timestep", "noise_prediction"}};
  std::shared_ptr<const SessionLoop> loop;
  EXPECT_FALSE(SessionLoop::Create(*session, options, loop).IsOK());

  auto model = stateful::MakeModel("incompatible_feedback_shapes");
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "left", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2});
  stateful::SetTensorType(*graph.add_input(), "right", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {3});
  stateful::SetTensorType(*graph.add_output(), "left_output", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2});
  stateful::SetTensorType(*graph.add_output(), "right_output", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {3});
  stateful::AddNode(graph, "Identity", {"left"}, "left_output");
  stateful::AddNode(graph, "Identity", {"right"}, "right_output");
  session = Session(model);
  options.output_names = {"left_output"};
  options.feedback = {{"left", "right_output"}};
  EXPECT_FALSE(SessionLoop::Create(*session, options, loop).IsOK());
}

TEST_P(SessionLoopTest, InvalidDynamicControlOutputsRequireReset) {
  auto session = Session(stateful::MakeFeedbackModel());
  auto options = Options();
  options.exit_conditions.push_back({LoopExitKind::BooleanOutput, 0, "done"});
  auto execution = Execution(*session, options, Inputs());
  session->after_run = [&](IOBinding& binding) {
    binding.GetOutputs()[1] = Value<bool>({1}, {true});
    return Status::OK();
  };
  SessionLoopStepResult result;
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());

  execution.reset();
  session = Session(TokenModel());
  options = {};
  options.output_names = {"token_output"};
  options.generated_token_output = "token_output";
  options.exit_conditions = {{LoopExitKind::MaxGeneratedTokens, 1}};
  execution = Execution(*session, options, {{{"token_input", Value<int64_t>({}, {1})}}, {}});
  session->after_run = [&](IOBinding& binding) {
    binding.GetOutputs()[0] = Value<int64_t>({1}, {1});
    return Status::OK();
  };
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_TRUE(execution->RequiresReset());
}

TEST(SessionLoopDefinitionTest, UninitializedAndStringSession) {
  SessionOptions options;
  InferenceSession unloaded(options, GetEnvironment());
  SessionLoopOptions config;
  config.output_names = {"output"};
  config.exit_conditions = {{LoopExitKind::MaxIterations, 1}};
  std::shared_ptr<const SessionLoop> loop;
  EXPECT_FALSE(SessionLoop::Create(unloaded, config, loop).IsOK());

  auto model = stateful::MakeModel("string_identity");
  auto& graph = *model.mutable_graph();
  stateful::SetTensorType(*graph.add_input(), "input", ONNX_NAMESPACE::TensorProto_DataType_STRING, {});
  stateful::SetTensorType(*graph.add_output(), "output", ONNX_NAMESPACE::TensorProto_DataType_STRING, {});
  stateful::AddNode(graph, "Identity", {"input"}, "output");
  const auto bytes = model.SerializeAsString();
  ASSERT_STATUS_OK(unloaded.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(unloaded.Initialize());
  EXPECT_FALSE(SessionLoop::Create(unloaded, config, loop).IsOK());
}

INSTANTIATE_TEST_SUITE_P(Cpu, SessionLoopTest, testing::Values(false));
#ifdef USE_CUDA
INSTANTIATE_TEST_SUITE_P(Cuda, SessionLoopTest, testing::Values(true));
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
