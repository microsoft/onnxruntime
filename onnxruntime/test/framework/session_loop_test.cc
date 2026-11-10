// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <future>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <limits>

#include "gtest/gtest.h"
#include "core/framework/tensor.h"
#include "core/graph/graph_viewer.h"
#include "core/session/inference_session.h"
#include "core/session/IOBinding.h"
#include "stateful/session/session_loop.h"
#include "stateful/session/session_chain.h"
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
  ASSERT_STATUS_OK(first->Reset());
  ASSERT_STATUS_OK(first->Step(result));
  EXPECT_EQ(result.iterations, 1);
  EXPECT_EQ(*result.outputs[0].Get<Tensor>().Data<float>(), 2.f);
  EXPECT_EQ(*retained[0].outputs[0].Get<Tensor>().Data<float>(), 2.f);
  EXPECT_EQ(*retained[1].outputs[0].Get<Tensor>().Data<float>(), 5.f);
  EXPECT_EQ(*retained[2].outputs[0].Get<Tensor>().Data<float>(), 9.f);
}

TEST_P(SessionLoopTest, DdimGuidanceReferenceSnapshotsCompletionAndReset) {
  for (const auto mode : {DdimGuidanceMode::Batched, DdimGuidanceMode::Sequential}) {
    for (const int64_t batch : {1, 2}) {
      auto policy = stateful::MakeDdimOptions(2);
      policy.guidance_mode = mode;
      policy.latent_shape[0] = batch;
      auto session = Session(stateful::MakeDdimDenoiserModel(policy));
      SessionLoopOptions options;
      options.ddim = policy;
      options.output_names = {policy.latent_output};
      options.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
      auto initial = batch == 1 ? Value<float>({1, 1, 1, 3}, {0.125f, -0.25f, 1.f})
                                : Value<float>({2, 1, 1, 3}, {0.125f, -0.25f, 1.f, 0.125f, -0.25f, 1.f});
      auto positive = batch == 1 ? Value<float>({1, 1, 1, 1}, {0.25f})
                                 : Value<float>({2, 1, 1, 1}, {0.25f, 0.25f});
      auto negative = batch == 1 ? Value<float>({1, 1, 1, 1}, {-0.5f})
                                 : Value<float>({2, 1, 1, 1}, {-0.5f, -0.5f});
      SessionLoopInputs inputs{{{"init", initial}, {"positive", positive}, {"negative", negative}}, {}};
      auto execution = Execution(*session, options, inputs);
      initial.GetMutable<Tensor>()->MutableData<float>()[0] = 99.f;
      SessionLoopStepResult first, final;
      ASSERT_STATUS_OK(execution->Step(first));
      EXPECT_TRUE(first.executed);
      EXPECT_FALSE(first.completed);
      ASSERT_EQ(first.outputs.size(), 1u);
      const auto retained = first.outputs[0];
      const auto first_value = retained.Get<Tensor>().Data<float>()[0];
      ASSERT_STATUS_OK(execution->Step(final));
      EXPECT_TRUE(final.completed);
      EXPECT_EQ(final.iterations, 2);
      EXPECT_EQ(final.invocations, mode == DdimGuidanceMode::Batched ? 2 : 4);
      EXPECT_EQ(session->calls, static_cast<size_t>(final.invocations));
      EXPECT_EQ(final.generated_tokens, 0);
      for (int64_t n = 0; n < batch; ++n) {
        const auto* data = final.outputs[0].Get<Tensor>().Data<float>() + n * 3;
        const std::array<float, 3> expected{-8.022068f, -8.661748f, -6.5294814f};
        for (size_t i = 0; i < expected.size(); ++i) {
          EXPECT_NEAR(data[i], expected[i], 1e-4f + 1e-4f * std::abs(expected[i]));
        }
      }
      ASSERT_STATUS_OK(execution->Step(final));
      EXPECT_FALSE(final.executed);
      EXPECT_EQ(session->calls, static_cast<size_t>(final.invocations));
      ASSERT_EQ(final.outputs.size(), 1u);
      ASSERT_STATUS_OK(execution->Reset());
      LoopFeeds state;
      ASSERT_STATUS_OK(execution->GetState(state));
      EXPECT_EQ(state.at(policy.latent_output).Get<Tensor>().Data<float>()[0], 0.125f);
      ASSERT_STATUS_OK(execution->Run(final));
      EXPECT_EQ(retained.Get<Tensor>().Data<float>()[0], first_value);
    }
  }
}

TEST_P(SessionLoopTest, DdimCancellationFinishesGuidanceBoundaryAndRequiresReset) {
  auto policy = stateful::MakeDdimOptions(3);
  policy.guidance_mode = DdimGuidanceMode::Sequential;
  auto session = Session(stateful::MakeDdimDenoiserModel(policy));
  SessionLoopOptions options;
  options.ddim = policy;
  options.output_names = {policy.latent_output};
  options.exit_conditions = {{LoopExitKind::MaxIterations, 3}};
  SessionLoopInputs inputs{{{"init", Value<uint64_t>({}, {0})},
                            {"positive", Value<float>({1, 1, 1, 1}, {0.25f})},
                            {"negative", Value<float>({1, 1, 1, 1}, {-0.5f})}},
                           {}};
  auto execution = Execution(*session, options, inputs);
  SessionLoopStepResult result;
  ASSERT_STATUS_OK(execution->Step(result));
  const auto retained = result.outputs[0];
  const float before = retained.Get<Tensor>().Data<float>()[0];
  session->before_run = [&] {
    execution->Cancel();
    return Status::OK();
  };
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_EQ(session->calls, 4u);
  EXPECT_TRUE(execution->RequiresReset());
  EXPECT_EQ(result.iterations, 1);
  EXPECT_EQ(retained.Get<Tensor>().Data<float>()[0], before);
  EXPECT_FALSE(execution->Step(result).IsOK());
  EXPECT_EQ(session->calls, 4u);
  session->before_run = {};
  ASSERT_STATUS_OK(execution->Reset());
  ASSERT_STATUS_OK(execution->Run(result));
  EXPECT_EQ(result.invocations, 6);
  EXPECT_EQ(retained.Get<Tensor>().Data<float>()[0], before);
}

TEST_P(SessionLoopTest, DdimEitherInvocationFailureDoesNotPublishAndResetReplays) {
  for (const size_t fail_call : {1u, 2u}) {
    auto policy = stateful::MakeDdimOptions(2);
    policy.guidance_mode = DdimGuidanceMode::Sequential;
    auto session = Session(stateful::MakeDdimDenoiserModel(policy));
    SessionLoopOptions options;
    options.ddim = policy;
    options.output_names = {policy.latent_output};
    options.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
    auto execution = Execution(*session, options, {{{"init", Value<uint64_t>({}, {123})}, {"positive", Value<float>({1, 1, 1, 1}, {0.25f})}, {"negative", Value<float>({1, 1, 1, 1}, {-0.5f})}}, {}});
    session->before_run = [&] {
      return session->calls == fail_call ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "controlled DDIM failure")
                                         : Status::OK();
    };
    SessionLoopStepResult result;
    EXPECT_FALSE(execution->Step(result).IsOK());
    EXPECT_TRUE(result.outputs.empty());
    EXPECT_TRUE(execution->RequiresReset());
    EXPECT_EQ(session->calls, fail_call);
    session->before_run = {};
    ASSERT_STATUS_OK(execution->Reset());
    ASSERT_STATUS_OK(execution->Run(result));
    const auto retained = result.outputs[0];
    ASSERT_STATUS_OK(execution->Reset());
    ASSERT_STATUS_OK(execution->Run(result));
    EXPECT_EQ(result.outputs[0].Get<Tensor>().Data<float>()[0], retained.Get<Tensor>().Data<float>()[0]);
  }
}

TEST_P(SessionLoopTest, DdimRejectsConflictingPoliciesMetadataAndLogicalInputs) {
  const auto policy = stateful::MakeDdimOptions(2);
  auto session = Session(stateful::MakeDdimDenoiserModel(policy));
  SessionLoopOptions valid;
  valid.ddim = policy;
  valid.output_names = {policy.latent_output};
  valid.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
  std::shared_ptr<const SessionLoop> loop;
  ASSERT_STATUS_OK(SessionLoop::Create(*session, valid, loop));
  auto reject = [&](SessionLoopOptions options) {
    auto output = loop;
    EXPECT_FALSE(SessionLoop::Create(*session, std::move(options), output).IsOK());
    EXPECT_EQ(output, loop);
  };
  auto options = valid;
  options.exit_conditions[0].limit = 0;
  reject(options);
  options = valid;
  options.output_names = {policy.prediction_output};
  reject(options);
  options = valid;
  options.feedback = {{policy.latent_input, policy.prediction_output}};
  reject(options);
  options = valid;
  options.ddim->latent_input = "missing";
  reject(options);
  options = valid;
  options.ddim->timestep_format = DdimTimestepFormat::Int64Vector;
  reject(options);
  options = valid;
  options.ddim->guidance_mode = DdimGuidanceMode::Sequential;
  reject(options);
  SessionLoopInputs inputs{{{"init", Value<int64_t>({}, {0})},
                            {"positive", Value<float>({1, 1, 1, 1}, {0.25f})},
                            {"negative", Value<float>({1, 1, 1, 1}, {-0.5f})}},
                           {}};
  std::unique_ptr<SessionLoopExecution> execution;
  EXPECT_FALSE(loop->CreateExecution(inputs, execution).IsOK());
  EXPECT_EQ(execution, nullptr);
  EXPECT_EQ(session->calls, 0u);
}

TEST_P(SessionLoopTest, DdimInvalidPredictionStopsAtTheFailingGuidanceInvocation) {
  for (const bool first_invalid : {false, true}) {
    auto policy = stateful::MakeDdimOptions(2);
    policy.guidance_mode = DdimGuidanceMode::Sequential;
    auto model = stateful::MakeDdimDenoiserModel(policy);
    auto& graph = *model.mutable_graph();
    graph.mutable_node(graph.node_size() - 1)->set_op_type("Div");
    auto session = Session(std::move(model));
    SessionLoopOptions options;
    options.ddim = policy;
    options.output_names = {policy.latent_output};
    options.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
    auto execution = Execution(*session, options,
                               {{{"init", Value<float>({1, 1, 1, 3}, {0.125f, -0.25f, 1.f})},
                                 {"positive", Value<float>({1, 1, 1, 1}, {first_invalid ? 0.25f : 0.f})},
                                 {"negative", Value<float>({1, 1, 1, 1}, {first_invalid ? 0.f : -0.5f})}},
                                {}});
    LoopFeeds initial;
    ASSERT_STATUS_OK(execution->GetState(initial));
    const auto retained = initial.at(policy.latent_output);
    SessionLoopStepResult result;
    EXPECT_FALSE(execution->Step(result).IsOK());
    EXPECT_TRUE(result.outputs.empty());
    EXPECT_EQ(session->calls, first_invalid ? 1u : 2u);
    EXPECT_TRUE(execution->RequiresReset());
    EXPECT_EQ(retained.Get<Tensor>().Data<float>()[0], 0.125f);
    EXPECT_FALSE(execution->GetState(initial).IsOK());
    ASSERT_STATUS_OK(execution->Reset());
    EXPECT_EQ(retained.Get<Tensor>().Data<float>()[0], 0.125f);
  }
}

TEST_P(SessionLoopTest, DdimSingleAndFiftyStepReferenceTrajectories) {
  const std::array<int64_t, 2> counts{1, 50};
  const std::array<std::array<float, 3>, 2> references{{{0.0626358911f, -0.3119553328f, 0.9366819859f},
                                                        {-50.66628265f, -53.9910202f, -42.90844727f}}};
  for (const auto mode : {DdimGuidanceMode::Batched, DdimGuidanceMode::Sequential}) {
    for (size_t case_index = 0; case_index < counts.size(); ++case_index) {
      auto policy = stateful::MakeDdimOptions(counts[case_index]);
      policy.guidance_mode = mode;
      auto session = Session(stateful::MakeDdimDenoiserModel(policy));
      SessionLoopOptions options;
      options.ddim = policy;
      options.output_names = {policy.latent_output};
      options.exit_conditions = {{LoopExitKind::MaxIterations, counts[case_index]}};
      auto execution = Execution(*session, options,
                                 {{{"init", Value<float>({1, 1, 1, 3}, {0.125f, -0.25f, 1.f})},
                                   {"positive", Value<float>({1, 1, 1, 1}, {0.25f})},
                                   {"negative", Value<float>({1, 1, 1, 1}, {-0.5f})}},
                                  {}});
      SessionLoopStepResult result;
      ASSERT_STATUS_OK(execution->Run(result));
      EXPECT_EQ(result.iterations, counts[case_index]);
      EXPECT_EQ(result.invocations, counts[case_index] * (mode == DdimGuidanceMode::Batched ? 1 : 2));
      EXPECT_EQ(session->calls, static_cast<size_t>(result.invocations));
      for (size_t i = 0; i < 3; ++i) {
        const float expected = references[case_index][i];
        EXPECT_NEAR(result.outputs[0].Get<Tensor>().Data<float>()[i], expected,
                    1e-4f + 1e-4f * std::abs(expected));
      }
    }
  }
}

TEST_P(SessionLoopTest, DdimAdditionalInvariantAndManagedPortValidation) {
  for (const auto mode : {DdimGuidanceMode::Batched, DdimGuidanceMode::Sequential}) {
    auto policy = stateful::MakeDdimOptions(2);
    policy.guidance_mode = mode;
    auto model = stateful::MakeDdimDenoiserModel(policy);
    auto& graph = *model.mutable_graph();
    stateful::SetTensorType(*graph.add_input(), "offset", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {});
    graph.mutable_node(graph.node_size() - 1)->set_output(0, "without_offset");
    stateful::AddNode(graph, "Add", {"without_offset", "offset"}, policy.prediction_output.c_str());
    auto session = Session(std::move(model));
    SessionLoopOptions options;
    options.ddim = policy;
    options.output_names = {policy.latent_output};
    options.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
    std::shared_ptr<const SessionLoop> loop;
    ASSERT_STATUS_OK(SessionLoop::Create(*session, options, loop));
    SessionLoopInputs inputs{{{"init", Value<float>({1, 1, 1, 3}, {0.125f, -0.25f, 1.f})},
                              {"positive", Value<float>({1, 1, 1, 1}, {0.25f})},
                              {"negative", Value<float>({1, 1, 1, 1}, {-0.5f})},
                              {"offset", Scalar(0.125f)}},
                             {}};
    std::unique_ptr<SessionLoopExecution> execution;
    ASSERT_STATUS_OK(loop->CreateExecution(inputs, execution));
    auto invalid = inputs;
    invalid.inputs.erase("offset");
    invalid.inputs.emplace(policy.latent_input, Scalar(0.f));
    const auto* original = execution.get();
    EXPECT_FALSE(loop->CreateExecution(invalid, execution).IsOK());
    EXPECT_EQ(execution.get(), original);
    SessionLoopStepResult result;
    ASSERT_STATUS_OK(execution->Run(result));
    const std::array<float, 3> expected{-8.22100067f, -8.86068058f, -6.72841311f};
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(result.outputs[0].Get<Tensor>().Data<float>()[i], expected[i],
                  1e-4f + 1e-4f * std::abs(expected[i]));
    }
  }
}

TEST_P(SessionLoopTest, DdimChainLogicalPortsStateHandoffAndActualInvocations) {
  for (const auto mode : {DdimGuidanceMode::Batched, DdimGuidanceMode::Sequential}) {
    for (const bool state_handoff : {false, true}) {
      auto policy = stateful::MakeDdimOptions(2);
      policy.guidance_mode = mode;
      auto denoiser = Session(stateful::MakeDdimDenoiserModel(policy));
      auto producer_model = stateful::MakeModel("ddim_conditioning_producer");
      auto& producer_graph = *producer_model.mutable_graph();
      stateful::SetTensorType(*producer_graph.add_input(), "prompt", ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                              {1, 1, 1, 1});
      stateful::SetTensorType(*producer_graph.add_output(), "embedding", ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                              {1, 1, 1, 1});
      stateful::AddNode(producer_graph, "Add", {"prompt", "prompt"}, "embedding");
      auto producer = Session(std::move(producer_model));
      auto consumer_model = stateful::MakeModel("ddim_latent_consumer");
      auto& consumer_graph = *consumer_model.mutable_graph();
      stateful::SetTensorType(*consumer_graph.add_input(), "incoming", ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                              {1, 1, 1, 3});
      stateful::SetTensorType(*consumer_graph.add_output(), "result", ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                              {1, 1, 1, 3});
      stateful::AddNode(consumer_graph, "Add", {"incoming", "incoming"}, "result");
      auto consumer = Session(std::move(consumer_model));
      SessionLoopOptions options;
      options.ddim = policy;
      options.output_names = {policy.latent_output};
      options.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
      std::shared_ptr<const SessionLoop> loop;
      ASSERT_STATUS_OK(SessionLoop::Create(*denoiser, options, loop));
      SessionChainOptions chain_options;
      SessionChainComponent conditioning;
      conditioning.name = "conditioning";
      conditioning.execution = producer.get();
      conditioning.inputs = {{"prompt", {ChainSourceKind::ExternalInput, "prompt"}}};
      chain_options.components.push_back(std::move(conditioning));
      SessionChainComponent diffusion;
      diffusion.name = "denoise";
      diffusion.execution = loop;
      diffusion.inputs = {{"init", {ChainSourceKind::ExternalInput, "init"}},
                          {"positive", {ChainSourceKind::ComponentOutput, "embedding", 0}},
                          {"negative", {ChainSourceKind::ExternalInput, "negative"}}};
      if (state_handoff) diffusion.state_outputs = {{"diffusion_latents", policy.latent_output}};
      chain_options.components.push_back(std::move(diffusion));
      SessionChainComponent output;
      output.name = "consume";
      output.execution = consumer.get();
      output.inputs = {{"incoming", state_handoff
                                        ? ChainValueSource{ChainSourceKind::State, "diffusion_latents"}
                                        : ChainValueSource{ChainSourceKind::ComponentOutput, policy.latent_output, 1}}};
      chain_options.components.push_back(std::move(output));
      chain_options.outputs = {{"final", 2, "result"}};
      std::shared_ptr<const SessionChain> chain;
      ASSERT_STATUS_OK(SessionChain::Create(chain_options, chain));
      SessionChainInputs inputs;
      inputs.inputs = {{"prompt", Value<float>({1, 1, 1, 1}, {0.125f})},
                       {"init", Value<float>({1, 1, 1, 3}, {0.125f, -0.25f, 1.f})},
                       {"negative", Value<float>({1, 1, 1, 1}, {-0.5f})}};
      std::unique_ptr<SessionChainExecution> execution;
      ASSERT_STATUS_OK(chain->CreateExecution(inputs, execution));
      SessionChainStepResult result;
      ASSERT_STATUS_OK(execution->Run(result));
      EXPECT_TRUE(result.completed);
      EXPECT_EQ(result.components_completed, 3u);
      EXPECT_EQ(result.invocations, mode == DdimGuidanceMode::Batched ? 4 : 6);
      EXPECT_EQ(static_cast<size_t>(result.invocations), producer->calls + denoiser->calls + consumer->calls);
      EXPECT_NEAR(result.outputs.at("final").Get<Tensor>().Data<float>()[0], -16.044136f,
                  1e-4f + 1e-4f * 16.044136f);
      const auto retained = result.outputs.at("final");
      if (state_handoff) {
        EXPECT_TRUE(execution->State().contains("diffusion_latents"));
      }
      ASSERT_STATUS_OK(execution->Reset());
      ASSERT_STATUS_OK(execution->Run(result));
      EXPECT_EQ(retained.Get<Tensor>().Data<float>()[0], result.outputs.at("final").Get<Tensor>().Data<float>()[0]);
    }
  }
}

TEST_P(SessionLoopTest, DdimChainInitializationUnionAndRawPredictionRejection) {
  const auto policy = stateful::MakeDdimOptions(2);
  auto denoiser = Session(stateful::MakeDdimDenoiserModel(policy));
  SessionLoopOptions options;
  options.ddim = policy;
  options.output_names = {policy.latent_output};
  options.exit_conditions = {{LoopExitKind::MaxIterations, 2}};
  std::shared_ptr<const SessionLoop> loop;
  ASSERT_STATUS_OK(SessionLoop::Create(*denoiser, options, loop));
  SessionChainOptions chain_options;
  SessionChainComponent component;
  component.name = "denoise";
  component.execution = loop;
  component.inputs = {{"init", {ChainSourceKind::State, "initialization"}},
                      {"positive", {ChainSourceKind::ExternalInput, "positive"}},
                      {"negative", {ChainSourceKind::ExternalInput, "negative"}}};
  chain_options.components.push_back(std::move(component));
  chain_options.outputs = {{"final", 0, policy.latent_output}};
  std::shared_ptr<const SessionChain> chain;
  ASSERT_STATUS_OK(SessionChain::Create(chain_options, chain));
  for (const auto& initial : {Value<uint64_t>({}, {std::numeric_limits<uint64_t>::max()}),
                              Value<float>({1, 1, 1, 3}, {0.125f, -0.25f, 1.f})}) {
    SessionChainInputs inputs;
    inputs.inputs = {{"positive", Value<float>({1, 1, 1, 1}, {0.25f})},
                     {"negative", Value<float>({1, 1, 1, 1}, {-0.5f})}};
    inputs.initial_state = {{"initialization", initial}};
    std::unique_ptr<SessionChainExecution> execution;
    ASSERT_STATUS_OK(chain->CreateExecution(inputs, execution));
    SessionChainStepResult result;
    ASSERT_STATUS_OK(execution->Run(result));
    const auto retained = result.outputs.at("final");
    ASSERT_STATUS_OK(execution->Reset());
    ASSERT_STATUS_OK(execution->Run(result));
    EXPECT_EQ(result.outputs.at("final").Get<Tensor>().Data<float>()[0], retained.Get<Tensor>().Data<float>()[0]);
  }
  chain_options.outputs[0].output_name = policy.prediction_output;
  auto unchanged = chain;
  EXPECT_FALSE(SessionChain::Create(chain_options, unchanged).IsOK());
  EXPECT_EQ(unchanged, chain);
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
  ASSERT_STATUS_OK(execution->Reset());

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
  ASSERT_STATUS_OK(execution->Reset());
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
  ASSERT_STATUS_OK(execution->Reset());
  session->before_run = []() -> Status { ORT_THROW("injected exception"); };
  EXPECT_THROW(ORT_THROW_IF_ERROR(execution->Step(result)), OnnxRuntimeException);
  EXPECT_TRUE(execution->RequiresReset());
  session->before_run = {};
  ASSERT_STATUS_OK(execution->Reset());
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
