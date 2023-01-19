// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <thread>
#include <random>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "core/framework/tensorprotoutils.h"
#include "orttraining/training_api/utils.h"
#include "orttraining/training_api/module.h"
#include "orttraining/training_api/optimizer.h"
#include "orttraining/training_api/checkpoint_property.h"
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/lr_scheduler.h"
#include "orttraining/test/training_api/core/data_utils.h"
#include "test/util/include/temp_dir.h"
#include "default_providers.h"

using json = nlohmann::json;

namespace onnxruntime {
namespace training {
namespace test {

namespace {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

void GenerateRandomData(std::vector<float>& data) {
  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;

  std::default_random_engine generator_float{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution_float{mean, scale};
  std::for_each(data.begin(), data.end(),
                [&generator_float, &distribution_float](float& value) { value = distribution_float(generator_float); });
}

void GenerateRandomInput(gsl::span<const int64_t> dims, OrtValue& input) {
  TensorShape shape(dims);
  std::vector<float> data(shape.Size());
  GenerateRandomData(data);
  onnxruntime::training::api::utils::CreateInputOrtValue<float>(dims, data, &input);
}

void TestModuleExport(const std::vector<std::shared_ptr<IExecutionProvider>>& providers) {
  auto training_model_uri = MODEL_FOLDER "training_model.onnx";
  auto eval_model_uri = MODEL_FOLDER "eval_model.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));
  auto model = std::make_unique<onnxruntime::training::api::Module>(
      ToUTF8String(training_model_uri), state.module_checkpoint_state.named_parameters, onnxruntime::SessionOptions(),
      *env, providers, ToUTF8String(eval_model_uri));

  auto test_dir = ORT_TSTR("export_model_for_inferencing_test_dir");
  if (Env::Default().FolderExists(test_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(test_dir).IsOK());
  }
  onnxruntime::test::TemporaryDirectory tmp_dir{test_dir};
  PathString inference_model_path{
      ConcatPathComponent<PathChar>(tmp_dir.Path(), ORT_TSTR("inference_model.onnx"))};

  std::vector<std::string> graph_output_names({"output-0"});
  ASSERT_STATUS_OK(model->ExportModelForInferencing(ToUTF8String(inference_model_path), graph_output_names));

  // Load model
  ONNX_NAMESPACE::ModelProto eval_model;
  ONNX_NAMESPACE::ModelProto inference_model;
  ORT_THROW_IF_ERROR(Model::Load(eval_model_uri, eval_model));
  ORT_THROW_IF_ERROR(Model::Load(inference_model_path, inference_model));

  // Check it has only one graph input
  ASSERT_EQ(eval_model.graph().input().size(), 6);
  ASSERT_EQ(inference_model.graph().input().size(), 1);
  ASSERT_EQ(inference_model.graph().input()[0].name(), "input-0");

  // Check that it does not have any node which has op type SoftmaxCrossEntropyLoss
  auto softmaxceloss_node_found = [](auto& model) -> bool {
    for (auto& node : model.graph().node()) {
      if (node.op_type() == "SoftmaxCrossEntropyLoss") {
        return true;
      }
    }
    return false;
  };
  ASSERT_EQ(softmaxceloss_node_found(eval_model), true);
  ASSERT_EQ(softmaxceloss_node_found(inference_model), false);

  // Try running an inference session
  auto inference_session = std::make_unique<onnxruntime::InferenceSession>(onnxruntime::SessionOptions(), *env);
  ASSERT_STATUS_OK(inference_session->Load(inference_model_path));
  ASSERT_STATUS_OK(inference_session->Initialize());
  std::vector<std::string> input_names({"input-0"});
  OrtValue graph_input;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, graph_input);
  std::vector<OrtValue> feeds;
  feeds.emplace_back(graph_input);
  std::vector<std::string> output_names({"output-0"});
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(inference_session->Run(RunOptions(), input_names, feeds, output_names, &outputs));
  ASSERT_EQ(outputs.size(), 1U);
}

#if defined(USE_CUDA)

const int64_t total_step_count = 100;
const float initial_lr = 1e-3f;
const int64_t resume_step = total_step_count / 2;

void CompareValue(float expected, float output, float rtol = 1e-4, float atol = 1e-5) {
  ASSERT_NEAR(expected, output, atol);
  ASSERT_NEAR(expected, output, rtol * std::abs(expected));
}

void TestLRSchduler(const std::basic_string<ORTCHAR_T>& test_file_name, float initial_lr, int64_t total_step_count,
                    int64_t warmup_step_count) {
  /// Load model and optimizer graph, create Module, Optimizer and LRScheduler instances.
  auto model_uri = MODEL_FOLDER "training_model.onnx";
  auto optim_uri = MODEL_FOLDER "adamw.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(LoadCheckpoint(checkpoint_to_load_path, state));

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));
  const std::vector<std::shared_ptr<IExecutionProvider>> providers{onnxruntime::test::DefaultCudaExecutionProvider()};
  auto model = std::make_unique<onnxruntime::training::api::Module>(
      ToUTF8String(model_uri), state.module_checkpoint_state.named_parameters,
      session_option, *env, providers);
  auto optim = std::make_shared<onnxruntime::training::api::Optimizer>(
      ToUTF8String(optim_uri), model->NamedParameters(), session_option,
      *env, providers);

  OrtValue input, target;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
  onnxruntime::training::api::utils::CreateInputOrtValue<int32_t>(
      std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1), &target);

  /// Load test data for learning rate schedulers.
  auto data_uri = ORT_TSTR("testdata/test_data_generation/lr_scheduler/" + test_file_name);
  std::ifstream in{data_uri};
  // Element of vector represent a pair of <step_count, list of learning rates>>
  typedef std::vector<std::pair<int64_t, std::vector<float>>> TestDataDictType;
  TestDataDictType test_data;
  const json j = json::parse(in);
  j.get_to<TestDataDictType>(test_data);

  int64_t resume_step = (*test_data.begin()).first;
  ASSERT_EQ(total_step_count, static_cast<int64_t>(test_data.size()) + resume_step);

  if (resume_step != 0) {
    /// Reset optimizer states to match the initial state we want to test.
    onnxruntime::training::api::OptimizerCheckpointState optimizer_checkpoint_states;
    auto group_opt_state =
        optimizer_checkpoint_states.group_named_optimizer_states["group0"] =
            std::make_shared<onnxruntime::training::api::GroupOptimizerState>();
    group_opt_state->step = resume_step;
    group_opt_state->initial_lr = initial_lr;
    ASSERT_STATUS_OK(optim->LoadStateDict(optimizer_checkpoint_states));
  }

  // KNOWN ISSUE: LinearLRScheduler by default use optim's states to calculate the first step's learning rate.
  // If we restored it after creation, it will only affect the learning rate from the second step.
  auto scheduler = std::make_unique<onnxruntime::training::api::LinearLRScheduler>(
      optim, warmup_step_count, total_step_count);

  for (auto it = test_data.begin(); it != test_data.end(); ++it) {
    onnxruntime::training::api::OptimizerCheckpointState optimizer_states;
    ASSERT_STATUS_OK(optim->GetStateDict(optimizer_states));
    auto group_optimizer_state = optimizer_states.group_named_optimizer_states["group0"];
    CompareValue(it->second[0], group_optimizer_state->learning_rate);
    ASSERT_EQ(it->first, group_optimizer_state->step);

    std::vector<OrtValue> inputs{input, target};
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
    ASSERT_STATUS_OK(optim->Step());
    ASSERT_STATUS_OK(scheduler->Step());
  }
}

#endif

}  // namespace

TEST(TrainingApiTest, ModuleParametersSize) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));
  auto model = std::make_unique<onnxruntime::training::api::Module>(ToUTF8String(model_uri),
                                                                    state.module_checkpoint_state.named_parameters, session_option,
                                                                    *env, std::vector<std::shared_ptr<IExecutionProvider>>());
  size_t params_size = 0;
  for (auto& param : model->Parameters()) {
    params_size += param->Data().Get<Tensor>().Shape().Size();
  }

  // ((500*784) + 500 + (10*500) + 10) = 397510
  ASSERT_EQ(params_size, 397510);
  ASSERT_EQ(model->GetParametersSize(), 397510);
}

TEST(TrainingApiTest, ModuleCopyBufferToParameters) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));
  auto model = std::make_unique<onnxruntime::training::api::Module>(ToUTF8String(model_uri),
                                                                    state.module_checkpoint_state.named_parameters, session_option,
                                                                    *env, std::vector<std::shared_ptr<IExecutionProvider>>());
  int64_t params_size = static_cast<int64_t>(model->GetParametersSize());
  std::vector<float> expected_param_buffer(params_size);
  GenerateRandomData(expected_param_buffer);

  OrtValue input_params;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),
                       {params_size},
                       reinterpret_cast<void*>(expected_param_buffer.data()),
                       onnxruntime::test::TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault)->Info(),
                       input_params, 0);
  ASSERT_STATUS_OK(model->CopyBufferToParameters(input_params));

  OrtValue output_params;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), {params_size},
                       onnxruntime::test::TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       output_params);
  ASSERT_STATUS_OK(model->CopyParametersToBuffer(output_params));

  const float* buffer = output_params.Get<Tensor>().Data<float>();
  ASSERT_TRUE(nullptr != buffer);
  for (int64_t i = 0; i < params_size; i++) {
    ASSERT_TRUE(*(buffer + i) == expected_param_buffer[i]);
  }
}

TEST(TrainingApiTest, ModuleTrainStep) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));
  auto model = std::make_unique<onnxruntime::training::api::Module>(ToUTF8String(model_uri),
                                                                    state.module_checkpoint_state.named_parameters, session_option,
                                                                    *env, std::vector<std::shared_ptr<IExecutionProvider>>());
  ASSERT_EQ(model->GetTrainingModelOutputCount(), 1);
  OrtValue input, target;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
  onnxruntime::training::api::utils::CreateInputOrtValue<int32_t>(
      std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1), &target);
  auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

  size_t step = 0;
  std::vector<float> single_bias_grad_vec, current_bias_grad_vec;
  std::string param_name = "fc2.weight";
  std::shared_ptr<onnxruntime::training::api::Parameter> bias_param = model->NamedParameters()[param_name];
  OrtValue& bias_grad = bias_param->Gradient();

  for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    step += 1;
    std::vector<OrtValue>& inputs = *it;
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
    ASSERT_EQ(fetches.size(), 1U);
    bias_grad = bias_param->Gradient();

    if (step > 1) {
      OrtValueToVec(bias_grad, current_bias_grad_vec);
      for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
        ASSERT_EQ(current_bias_grad_vec[i], single_bias_grad_vec[i] * step);
      }
    } else {
      OrtValueToVec(bias_grad, single_bias_grad_vec);
    }
  }
  // reset grad
  ASSERT_STATUS_OK(model->LazyResetGrad());

  // run a single step
  std::vector<OrtValue>& inputs = *data_loader.begin();
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
  OrtValueToVec(bias_grad, current_bias_grad_vec);
  for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
    ASSERT_EQ(current_bias_grad_vec[i], single_bias_grad_vec[i]);
  }
}

TEST(TrainingApiTest, ModuleExportModelForInferencingCPU) {
  std::vector<std::shared_ptr<IExecutionProvider>> providers{onnxruntime::test::DefaultCpuExecutionProvider()};
  TestModuleExport(providers);
}

#if defined(USE_CUDA)

TEST(TrainingApiTest, ModuleExportModelForInferencingCUDA) {
  std::vector<std::shared_ptr<IExecutionProvider>> providers{onnxruntime::test::DefaultCudaExecutionProvider()};
  TestModuleExport(providers);
}

TEST(TrainingApiTest, OptimStep) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";
  auto optim_uri = MODEL_FOLDER "adamw.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  std::vector<std::shared_ptr<IExecutionProvider>> providers{onnxruntime::test::DefaultCudaExecutionProvider()};
  std::shared_ptr<IExecutionProvider> cuda_provider = providers.front();
  std::shared_ptr<IExecutionProvider> cpu_provider = onnxruntime::test::DefaultCpuExecutionProvider();
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));
  auto model = std::make_unique<onnxruntime::training::api::Module>(
      ToUTF8String(model_uri), state.module_checkpoint_state.named_parameters, session_option,
      *env, providers);
  auto optim = std::make_unique<onnxruntime::training::api::Optimizer>(
      ToUTF8String(optim_uri), model->NamedParameters(), session_option,
      *env, providers);

  OrtValue input, target;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
  onnxruntime::training::api::utils::CreateInputOrtValue<int32_t>(
      std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1), &target);
  auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

  size_t step = 0;
  std::string param_name = "fc2.weight";

  // before training, check if optim state is initialized to 0
  onnxruntime::training::api::OptimizerCheckpointState optimizer_states;
  ASSERT_STATUS_OK(optim->GetStateDict(optimizer_states));
  onnxruntime::training::api::ParameterOptimizerState& param_state =
      optimizer_states.group_named_optimizer_states["group0"]->param_named_optimizer_states.at(param_name);
  OrtValue& moment_1 = param_state.momentum_named_states.at("momentum0");

  std::vector<float> param_vec_before_optimizer_step;
  CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Data(), param_vec_before_optimizer_step,
                       cuda_provider, cpu_provider);
  std::vector<float> moment_1_vec;
  CudaOrtValueToCpuVec(moment_1, moment_1_vec, cuda_provider, cpu_provider);
  for (size_t i = 0; i < moment_1_vec.size(); i++) {
    ASSERT_EQ(moment_1_vec[i], 0.0f);
  }

  for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    step += 1;
    std::vector<OrtValue>& inputs = *it;
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
    std::vector<float> grads;
    CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Gradient(), grads,
                         cuda_provider, cpu_provider);
    ASSERT_STATUS_OK(optim->Step());

    // get optim state and check if it is updated
    CudaOrtValueToCpuVec(moment_1, moment_1_vec, cuda_provider, cpu_provider);
    for (size_t i = 0; i < moment_1_vec.size(); i++) {
      if (grads[i] != 0.0f) {
        ASSERT_NE(moment_1_vec[i], 0.0f);
      }
    }

    std::vector<float> param_vec_after_optimizer_step;
    CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Data(), param_vec_after_optimizer_step,
                         cuda_provider, cpu_provider);
    for (size_t i = 0; i < param_vec_after_optimizer_step.size(); ++i) {
      if (grads[i] != 0.0f && moment_1_vec[i] != 0.0f) {
        ASSERT_NE(param_vec_after_optimizer_step[i], param_vec_before_optimizer_step[i]);
      }
    }
  }
}

TEST(TrainingApiTest, LinearLRScheduler_NoWarmUp_Test) {
  // No warm up.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-0.json"), initial_lr, total_step_count, 0);
}

TEST(TrainingApiTest, LinearLRScheduler_NoWarmUp_ResumeFromCheckpoint_Test) {
  // No warm up.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-0_restored.json"), initial_lr, total_step_count, 0);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp30Step_Test) {
  // Warmp up completed before saving checkpoint.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-30.json"), initial_lr, total_step_count, 30);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp30Step_ResumeFromCheckpoint_Test) {
  // Warmp up completed before saving checkpoint.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-30_restored.json"), initial_lr, total_step_count, 30);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp70Step_Test) {
  // Warmp up completed after saving checkpoint.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-70.json"), initial_lr, total_step_count, 70);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp70Step_ResumeFromCheckpoint_Test) {
  // Warmp up completed after saving checkpoint.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-70_restored.json"), initial_lr, total_step_count, 70);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp200Step_Test) {
  // All steps are in warm-up phase.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-200.json"), initial_lr, total_step_count, 200);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp200Step_ResumeFromCheckpoint_Test) {
  // All steps are in warm-up phase.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-200_restored.json"), initial_lr, total_step_count, 200);
}

#endif

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
