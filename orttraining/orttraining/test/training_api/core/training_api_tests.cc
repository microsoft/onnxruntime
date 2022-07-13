// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <thread>
#include <random>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "core/framework/tensorprotoutils.h"
#include "orttraining/training_api/include/utils.h"
#include "orttraining/training_api/include/module.h"
#include "orttraining/training_api/include/optimizer.h"
#include "orttraining/training_api/include/checkpoint_property.h"
#include "orttraining/training_api/include/checkpoint.h"
#include "orttraining/training_api/include/lr_scheduler.h"
#include "orttraining/test/training_api/core/data_utils.h"
#include "default_providers.h"

using json = nlohmann::json;

namespace onnxruntime {
namespace training {
namespace test {

namespace {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

void GenerateRandomInput(gsl::span<const int64_t> dims, OrtValue& input) {
  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;

  TensorShape shape(dims);
  std::default_random_engine generator_float{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution_float{mean, scale};
  std::vector<float> data(shape.Size());
  std::for_each(data.begin(), data.end(),
                [&generator_float, &distribution_float](float& value) { value = distribution_float(generator_float); });
  onnxruntime::training::api::utils::CreateInputOrtValue<float>(dims, data, &input);
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
  ASSERT_EQ(model->GetTrainModeOutputCount(), 1);
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
  ASSERT_STATUS_OK(model->ResetGrad());

  // run a single step
  std::vector<OrtValue>& inputs = *data_loader.begin();
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
  OrtValueToVec(bias_grad, current_bias_grad_vec);
  for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
    ASSERT_EQ(current_bias_grad_vec[i], single_bias_grad_vec[i]);
  }
}

#if defined(USE_CUDA) || defined(USE_ROCM)

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

void CompareValue(float expected, float output, float rtol = 1e-4, float atol = 1e-5) {
  ASSERT_NEAR(expected, output, atol);
  ASSERT_NEAR(expected, output, rtol * std::abs(expected));
}

void TestLRSchduler(const std::string& test_file_name, float initial_lr, int64_t total_step_count,
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

const int64_t total_step_count = 100;
const float initial_lr = 1e-3f;
const int64_t resume_step = total_step_count / 2;
TEST(TrainingApiTest, LinearLRScheduler_NoWarmUp_Test) {
  // No warm up.
  TestLRSchduler("warmup_linear_scheduler_warmupstep-0.json", initial_lr, total_step_count, 0);
}

TEST(TrainingApiTest, LinearLRScheduler_NoWarmUp_ResumeFromCheckpoint_Test) {
  // No warm up.
  TestLRSchduler("warmup_linear_scheduler_warmupstep-0_restored.json", initial_lr, total_step_count, 0);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp30Step_Test) {
  // Warmp up completed before saving checkpoint.
  TestLRSchduler("warmup_linear_scheduler_warmupstep-30.json", initial_lr, total_step_count, 30);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp30Step_ResumeFromCheckpoint_Test) {
  // Warmp up completed before saving checkpoint.
  TestLRSchduler("warmup_linear_scheduler_warmupstep-30_restored.json", initial_lr, total_step_count, 30);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp70Step_Test) {
  // Warmp up completed after saving checkpoint.
  TestLRSchduler("warmup_linear_scheduler_warmupstep-70.json", initial_lr, total_step_count, 70);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp70Step_ResumeFromCheckpoint_Test) {
  // Warmp up completed after saving checkpoint.
  TestLRSchduler("warmup_linear_scheduler_warmupstep-70_restored.json", initial_lr, total_step_count, 70);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp200Step_Test) {
  // All steps are in warm-up phase.
  TestLRSchduler("warmup_linear_scheduler_warmupstep-200.json", initial_lr, total_step_count, 200);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp200Step_ResumeFromCheckpoint_Test) {
  // All steps are in warm-up phase.
  TestLRSchduler("warmup_linear_scheduler_warmupstep-200_restored.json", initial_lr, total_step_count, 200);
}

#endif

}  // namespace
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
