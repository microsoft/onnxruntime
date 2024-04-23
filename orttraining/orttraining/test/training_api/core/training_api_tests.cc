// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <thread>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

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
using namespace onnxruntime::training::api;

namespace onnxruntime {
namespace training {
namespace test {

namespace {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

constexpr int64_t TOTAL_STEP_COUNT = 100;
constexpr float INITIAL_LR = 1e-3f;

/**
 * @brief Create a Fake Optimizer Checkpoint State On CPU.
 *
 * @param named_parameters Parameter list
 * @param momentum_keys Optimizer momentum keys.
 * @param optimizer_checkpoint_state Used as output to store the state containing faked data.
 * @return Status
 */
Status CreateFakeOptimizerCheckpointStateOnCPU(
    const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
    const InlinedVector<std::string>& momentum_keys,
    OptimizerCheckpointState& optimizer_checkpoint_state) {
  auto& grouped_optimizer_states = optimizer_checkpoint_state.group_named_optimizer_states;
  grouped_optimizer_states.insert({"group0", std::make_shared<GroupOptimizerState>()});
  GroupOptimizerState& group_optimizer_state = *(grouped_optimizer_states["group0"]);

  auto& param_named_optimizer_states = group_optimizer_state.param_named_optimizer_states;
  for (auto& pair : named_parameters) {
    if (pair.second->RequiresGrad()) {
      param_named_optimizer_states.insert({pair.first, ParameterOptimizerState()});
      ParameterOptimizerState& cur_param_optimizer_states = param_named_optimizer_states[pair.first];
      for (auto& state_name : momentum_keys) {
        OrtValue param_moment_state;
        OrtValue param = pair.second->Data();
        const auto& param_tensor = param.template Get<Tensor>();
        GenerateRandomInput(param_tensor.Shape().GetDims(), param_moment_state);
        cur_param_optimizer_states.insert({state_name, std::move(param_moment_state)});
      }
    }
  }

  return Status::OK();
}

void TestModuleExport(const std::vector<std::shared_ptr<IExecutionProvider>>& providers) {
  auto training_model_uri = MODEL_FOLDER "training_model.onnx";
  auto eval_model_uri = MODEL_FOLDER "eval_model.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));
  auto model_identifier = ModelIdentifiers(onnxruntime::ToUTF8String(training_model_uri),
                                           std::optional<std::string>(onnxruntime::ToUTF8String(eval_model_uri)),
                                           std::nullopt);
  auto model = std::make_unique<onnxruntime::training::api::Module>(
      model_identifier, &state, onnxruntime::SessionOptions(),
      *env, providers);

  auto test_dir = ORT_TSTR("export_model_for_inferencing_test_dir");
  if (Env::Default().FolderExists(test_dir)) {
    ORT_ENFORCE(Env::Default().DeleteFolder(test_dir).IsOK());
  }
  onnxruntime::test::TemporaryDirectory tmp_dir{test_dir};
  PathString inference_model_path{
      ConcatPathComponent(tmp_dir.Path(), ORT_TSTR("inference_model.onnx"))};

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

}  // namespace

TEST(TrainingApiTest, ModuleParametersSize) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));
  auto model_identifiers = ModelIdentifiers(onnxruntime::ToUTF8String(model_uri),
                                            std::nullopt, std::nullopt);
  auto model = std::make_unique<onnxruntime::training::api::Module>(model_identifiers,
                                                                    &state, session_option,
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
  auto model_identifier = ModelIdentifiers(onnxruntime::ToUTF8String(model_uri),
                                           std::nullopt,
                                           std::nullopt);
  auto model = std::make_unique<onnxruntime::training::api::Module>(model_identifier,
                                                                    &state, session_option,
                                                                    *env, std::vector<std::shared_ptr<IExecutionProvider>>());
  int64_t params_size = static_cast<int64_t>(model->GetParametersSize());
  std::vector<float> expected_param_buffer(params_size);
  GenerateRandomData(expected_param_buffer);

  OrtValue input_params;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),
                       {params_size},
                       reinterpret_cast<void*>(expected_param_buffer.data()),
                       onnxruntime::test::TestCPUExecutionProvider()->CreatePreferredAllocators()[0]->Info(),
                       input_params, 0);
  ASSERT_STATUS_OK(model->CopyBufferToParameters(input_params));

  OrtValue output_params;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), {params_size},
                       onnxruntime::test::TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
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
  auto model_identifier = ModelIdentifiers(onnxruntime::ToUTF8String(model_uri),
                                           std::nullopt,
                                           std::nullopt);
  auto model = std::make_unique<onnxruntime::training::api::Module>(model_identifier,
                                                                    &state, session_option,
                                                                    *env, std::vector<std::shared_ptr<IExecutionProvider>>());
  ASSERT_EQ(model->GetTrainingModelOutputCount(), 1);
  OrtValue input, target;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
  target = onnxruntime::test::CreateInputOrtValueOnCPU<int32_t>(
      std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1));
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
      CpuOrtValueToVec(bias_grad, current_bias_grad_vec);
      for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
        ASSERT_EQ(current_bias_grad_vec[i], single_bias_grad_vec[i] * step);
      }
    } else {
      CpuOrtValueToVec(bias_grad, single_bias_grad_vec);
    }
  }
  // reset grad
  ASSERT_STATUS_OK(model->LazyResetGrad());

  // run a single step
  std::vector<OrtValue>& inputs = *data_loader.begin();
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
  CpuOrtValueToVec(bias_grad, current_bias_grad_vec);
  for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
    ASSERT_EQ(current_bias_grad_vec[i], single_bias_grad_vec[i]);
  }
}

TEST(TrainingApiTest, OptimizerCreatedWithOptimizerCheckpointState) {
  std::vector<bool> run_cuda_list{false};
  // #ifdef USE_CUDA
  //   run_cuda_list.push_back(true);
  // #endif

  for (auto run_cuda : run_cuda_list) {
    std::vector<std::shared_ptr<IExecutionProvider>> providers;
    if (run_cuda) {
      providers = {onnxruntime::test::DefaultCudaExecutionProvider()};
    } else {
      providers = {onnxruntime::test::DefaultCpuExecutionProvider()};
    }

    auto model_uri = MODEL_FOLDER "training_model.onnx";
    auto optim_uri = MODEL_FOLDER "adamw.onnx";

    CheckpointState state;
    auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
    ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

    onnxruntime::SessionOptions session_option;
    std::unique_ptr<Environment> env;

    ASSERT_STATUS_OK(Environment::Create(nullptr, env));

    auto model_identifier = ModelIdentifiers(onnxruntime::ToUTF8String(model_uri),
                                             std::nullopt,
                                             std::optional<std::string>(onnxruntime::ToUTF8String(optim_uri)));

    std::shared_ptr<Module> model = std::make_shared<Module>(
        model_identifier, &state, session_option,
        *env, providers);

    // Load state dict from faked optimizer checkpoint state.
    CheckpointState new_state = state;
    OptimizerCheckpointState& external_optimizer_checkpoint_state = new_state.optimizer_checkpoint_state;
    ASSERT_STATUS_OK(CreateFakeOptimizerCheckpointStateOnCPU(model->NamedParameters(),
                                                             {"momentum0", "momentum1"},
                                                             external_optimizer_checkpoint_state));
    std::shared_ptr<Optimizer> optim = std::make_shared<Optimizer>(
        model_identifier, &new_state, session_option, *env, providers);

    ASSERT_TRUE(optim.get() != nullptr);
  }
}

void TestLRSchduler(const std::basic_string<ORTCHAR_T>& test_file_name,
                    float initial_lr,
                    int64_t total_step_count,
                    int64_t warmup_step_count) {
  std::vector<bool> run_cuda_list{false};
#ifdef USE_CUDA
  run_cuda_list.push_back(true);
#endif

  for (auto run_cuda : run_cuda_list) {
    std::vector<std::shared_ptr<IExecutionProvider>> providers;
    if (run_cuda) {
      providers = {onnxruntime::test::DefaultCudaExecutionProvider()};
    } else {
      providers = {onnxruntime::test::DefaultCpuExecutionProvider()};
    }

    auto model_uri = MODEL_FOLDER "training_model.onnx";
    auto optim_uri = MODEL_FOLDER "adamw.onnx";

    CheckpointState state;
    auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
    ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

    onnxruntime::SessionOptions session_option;
    std::unique_ptr<Environment> env;

    ASSERT_STATUS_OK(Environment::Create(nullptr, env));

    auto model_identifier = ModelIdentifiers(onnxruntime::ToUTF8String(model_uri),
                                             std::nullopt,
                                             std::optional<std::string>(onnxruntime::ToUTF8String(optim_uri)));

    std::shared_ptr<Module> model = std::make_shared<Module>(
        model_identifier, &state, session_option,
        *env, providers);

    OrtValue input, target;
    GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
    target = onnxruntime::test::CreateInputOrtValueOnCPU<int32_t>(
        std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1));

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
      state.optimizer_checkpoint_state.group_named_optimizer_states.insert(
          {"group0", std::make_shared<GroupOptimizerState>()});
      auto& group_opt_state = state.optimizer_checkpoint_state.group_named_optimizer_states["group0"];
      /// Reset optimizer states to match the initial state we want to test.
      group_opt_state->step = resume_step;
      group_opt_state->initial_lr = initial_lr;
    }

    std::shared_ptr<Optimizer> optim = std::make_shared<Optimizer>(
        model_identifier, &state, session_option,
        *env, providers);

    // KNOWN ISSUE: LinearLRScheduler by default use optim's states to calculate the first step's learning rate.
    // If we restored it after creation, it will only affect the learning rate from the second step.
    auto scheduler = std::make_unique<LinearLRScheduler>(
        optim, warmup_step_count, total_step_count);

    for (auto it = test_data.begin(); it != test_data.end(); ++it) {
      OptimizerCheckpointState& optimizer_states = state.optimizer_checkpoint_state;
      auto group_optimizer_state = optimizer_states.group_named_optimizer_states["group0"];

      constexpr const float rtol = 1e-4f, atol = 1e-5f;
      ASSERT_NEAR(it->second[0], group_optimizer_state->learning_rate, atol);
      ASSERT_NEAR(it->second[0], group_optimizer_state->learning_rate, rtol * std::abs(it->second[0]));

      ASSERT_EQ(it->first, group_optimizer_state->step);

      std::vector<OrtValue> inputs{input, target};
      std::vector<OrtValue> fetches;
      ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
      ASSERT_STATUS_OK(optim->Step());
      ASSERT_STATUS_OK(scheduler->Step());
    }
  }
}

TEST(TrainingApiTest, LinearLRScheduler_NoWarmUp_Test) {
  // No warm up.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-0.json"), INITIAL_LR, TOTAL_STEP_COUNT, 0);
}

TEST(TrainingApiTest, LinearLRScheduler_NoWarmUp_ResumeFromCheckpoint_Test) {
  // No warm up.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-0_restored.json"), INITIAL_LR, TOTAL_STEP_COUNT, 0);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp30Step_Test) {
  // Warmp up completed before saving checkpoint.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-30.json"), INITIAL_LR, TOTAL_STEP_COUNT, 30);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp30Step_ResumeFromCheckpoint_Test) {
  // Warmp up completed before saving checkpoint.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-30_restored.json"), INITIAL_LR, TOTAL_STEP_COUNT, 30);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp70Step_Test) {
  // Warmp up completed after saving checkpoint.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-70.json"), INITIAL_LR, TOTAL_STEP_COUNT, 70);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp70Step_ResumeFromCheckpoint_Test) {
  // Warmp up completed after saving checkpoint.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-70_restored.json"), INITIAL_LR, TOTAL_STEP_COUNT, 70);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp200Step_Test) {
  // All steps are in warm-up phase.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-200.json"), INITIAL_LR, TOTAL_STEP_COUNT, 200);
}

TEST(TrainingApiTest, LinearLRScheduler_WarmUp200Step_ResumeFromCheckpoint_Test) {
  // All steps are in warm-up phase.
  TestLRSchduler(ORT_TSTR("warmup_linear_scheduler_warmupstep-200_restored.json"), INITIAL_LR, TOTAL_STEP_COUNT, 200);
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
#endif

TEST(TrainingApiTest, OptimStep) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";
  auto optim_uri = MODEL_FOLDER "adamw.onnx";

  onnxruntime::training::api::CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  std::vector<std::shared_ptr<IExecutionProvider>> providers;
#if defined(USE_CUDA)
  providers.push_back(onnxruntime::test::DefaultCudaExecutionProvider());
#endif
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));

  auto model_identifier = ModelIdentifiers(onnxruntime::ToUTF8String(model_uri),
                                           std::nullopt,
                                           std::optional<std::string>(onnxruntime::ToUTF8String(optim_uri)));
  auto model = std::make_unique<onnxruntime::training::api::Module>(
      model_identifier, &state, session_option,
      *env, providers);
  auto optim = std::make_unique<onnxruntime::training::api::Optimizer>(
      model_identifier, &state, session_option,
      *env, providers);

  OrtValue input, target;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
  target = onnxruntime::test::CreateInputOrtValueOnCPU<int32_t>(
      std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1));
  auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

  std::string param_name = "fc2.weight";
  // before training, check if optim state is initialized to 0
  onnxruntime::training::api::OptimizerCheckpointState& optimizer_states = state.optimizer_checkpoint_state;
  onnxruntime::training::api::ParameterOptimizerState& param_state =
      optimizer_states.group_named_optimizer_states["group0"]->param_named_optimizer_states.at(param_name);
  OrtValue& moment_1 = param_state.at("momentum0");

  std::vector<float> param_vec_before_optimizer_step;
  std::vector<float> moment_1_vec;
#if defined(USE_CUDA)
  CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Data(), param_vec_before_optimizer_step);
  CudaOrtValueToCpuVec(moment_1, moment_1_vec);
#else
  CpuOrtValueToVec(model->NamedParameters().at(param_name)->Data(), param_vec_before_optimizer_step);
  CpuOrtValueToVec(moment_1, moment_1_vec);
#endif

  for (size_t i = 0; i < moment_1_vec.size(); i++) {
    ASSERT_EQ(moment_1_vec[i], 0.0f);
  }

  for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    std::vector<OrtValue>& inputs = *it;
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
    ASSERT_STATUS_OK(optim->Step());

    // get gradients and optim state and check if it is updated
    std::vector<float> grads;
#if defined(USE_CUDA)
    CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Gradient(), grads);
    CudaOrtValueToCpuVec(moment_1, moment_1_vec);
#else
    CpuOrtValueToVec(model->NamedParameters().at(param_name)->Gradient(), grads);
    CpuOrtValueToVec(moment_1, moment_1_vec);
#endif
    for (size_t i = 0; i < moment_1_vec.size(); i++) {
      if (grads[i] != 0.0f) {
        ASSERT_NE(moment_1_vec[i], 0.0f);
      }
    }

    std::vector<float> param_vec_after_optimizer_step;
#if defined(USE_CUDA)
    CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Data(), param_vec_after_optimizer_step);
#else
    CpuOrtValueToVec(model->NamedParameters().at(param_name)->Data(), param_vec_after_optimizer_step);
#endif
    for (size_t i = 0; i < param_vec_after_optimizer_step.size(); ++i) {
      if (grads[i] != 0.0f && moment_1_vec[i] != 0.0f) {
        ASSERT_NE(param_vec_after_optimizer_step[i], param_vec_before_optimizer_step[i]);
      }
    }
  }
}

TEST(TrainingApiTest, ModuleAndOptimizerWithNominalState) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";
  auto eval_model_uri = MODEL_FOLDER "eval_model.onnx";
  auto optim_uri = MODEL_FOLDER "adamw.onnx";

  onnxruntime::training::api::CheckpointState complete_state;
  onnxruntime::training::api::CheckpointState nominal_state;
  auto complete_checkpoint_path = MODEL_FOLDER "checkpoint.ckpt";
  auto nominal_checkpoint_path = MODEL_FOLDER "nominal_checkpoint";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(complete_checkpoint_path, complete_state));
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(nominal_checkpoint_path, nominal_state));

  ASSERT_FALSE(complete_state.module_checkpoint_state.is_nominal_state);
  ASSERT_TRUE(nominal_state.module_checkpoint_state.is_nominal_state);

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  std::vector<std::shared_ptr<IExecutionProvider>> providers;
#if defined(USE_CUDA)
  providers.push_back(onnxruntime::test::DefaultCudaExecutionProvider());
#endif
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));

  auto model_identifier = ModelIdentifiers(onnxruntime::ToUTF8String(model_uri),
                                           std::optional<std::string>(onnxruntime::ToUTF8String(eval_model_uri)),
                                           std::optional<std::string>(onnxruntime::ToUTF8String(optim_uri)));
  auto model_with_complete_state = std::make_unique<onnxruntime::training::api::Module>(
      model_identifier, &complete_state, session_option,
      *env, providers);
  auto model_with_nominal_state = std::make_unique<onnxruntime::training::api::Module>(
      model_identifier, &nominal_state, session_option,
      *env, providers);
  auto optim_with_complete_state = std::make_unique<onnxruntime::training::api::Optimizer>(
      model_identifier, &complete_state, session_option,
      *env, providers);
  auto optim_with_nominal_state = std::make_unique<onnxruntime::training::api::Optimizer>(
      model_identifier, &nominal_state, session_option,
      *env, providers);

  // Before running the test, copy all the parameters to the nominal module.
  ASSERT_EQ(model_with_complete_state->GetParametersSize(), model_with_nominal_state->GetParametersSize());
  int64_t params_size = static_cast<int64_t>(model_with_nominal_state->GetParametersSize());
  OrtValue params_buffer;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), {params_size},
                       onnxruntime::test::TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                       params_buffer);
  ASSERT_STATUS_OK(model_with_complete_state->CopyParametersToBuffer(params_buffer, false));
  ASSERT_STATUS_OK(model_with_nominal_state->CopyBufferToParameters(params_buffer, false));

  ASSERT_STATUS_OK(optim_with_nominal_state->ConstructOptimizerStateAndInputs());

  OrtValue input, target;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
  target = onnxruntime::test::CreateInputOrtValueOnCPU<int32_t>(
      std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1));
  auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

  for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    std::vector<OrtValue>& inputs = *it;
    std::vector<OrtValue> complete_fetches;
    std::vector<OrtValue> nominal_fetches;
    ASSERT_STATUS_OK(model_with_complete_state->TrainStep(inputs, complete_fetches));
    ASSERT_STATUS_OK(model_with_nominal_state->TrainStep(inputs, nominal_fetches));

    ASSERT_GT(complete_fetches.size(), 0);
    for (size_t i = 0; i < complete_fetches.size(); ++i) {
      ASSERT_TRUE(complete_fetches[i].IsTensor());
      ASSERT_TRUE(nominal_fetches[i].IsTensor());
      const Tensor& complete_tensor = complete_fetches[i].Get<Tensor>();
      const Tensor& nominal_tensor = nominal_fetches[i].Get<Tensor>();
      ASSERT_EQ(complete_tensor.Shape(), nominal_tensor.Shape());
      ASSERT_EQ(complete_tensor.DataType(), nominal_tensor.DataType());

      std::vector<float> complete_fetches_vec;
      std::vector<float> nominal_fetches_vec;
#if defined(USE_CUDA)
      CudaOrtValueToCpuVec(complete_fetches[i], complete_fetches_vec);
      CudaOrtValueToCpuVec(nominal_fetches[i], nominal_fetches_vec);
#else
      CpuOrtValueToVec(complete_fetches[i], complete_fetches_vec);
      CpuOrtValueToVec(nominal_fetches[i], nominal_fetches_vec);
#endif

      for (size_t j = 0; j < complete_fetches_vec.size(); ++j) {
        ASSERT_EQ(complete_fetches_vec[j], nominal_fetches_vec[j]);
      }
    }

    ASSERT_STATUS_OK(optim_with_complete_state->Step());
    ASSERT_STATUS_OK(optim_with_nominal_state->Step());

    for (auto& [name, param] : model_with_complete_state->NamedParameters()) {
      ASSERT_TRUE(param->Data().IsTensor());
      ASSERT_TRUE(param->Gradient().IsTensor());
      ASSERT_TRUE(model_with_nominal_state->NamedParameters().at(name)->Data().IsTensor());
      ASSERT_TRUE(model_with_nominal_state->NamedParameters().at(name)->Gradient().IsTensor());

      const Tensor& complete_data = param->Data().Get<Tensor>();
      const Tensor& complete_grad = param->Gradient().Get<Tensor>();
      const Tensor& nominal_data = model_with_nominal_state->NamedParameters().at(name)->Data().Get<Tensor>();
      const Tensor& nominal_grad = model_with_nominal_state->NamedParameters().at(name)->Gradient().Get<Tensor>();

      ASSERT_EQ(complete_data.Shape(), nominal_data.Shape());
      ASSERT_EQ(complete_data.DataType(), nominal_data.DataType());
      ASSERT_EQ(complete_grad.Shape(), nominal_grad.Shape());
      ASSERT_EQ(complete_grad.DataType(), nominal_grad.DataType());

      std::vector<float> complete_data_vec;
      std::vector<float> complete_grad_vec;
      std::vector<float> nominal_data_vec;
      std::vector<float> nominal_grad_vec;

#if defined(USE_CUDA)
      CudaOrtValueToCpuVec(param->Data(), complete_data_vec);
      CudaOrtValueToCpuVec(param->Gradient(), complete_grad_vec);
      CudaOrtValueToCpuVec(model_with_nominal_state->NamedParameters().at(name)->Data(), nominal_data_vec);
      CudaOrtValueToCpuVec(model_with_nominal_state->NamedParameters().at(name)->Gradient(), nominal_grad_vec);
#else
      CpuOrtValueToVec(param->Data(), complete_data_vec);
      CpuOrtValueToVec(param->Gradient(), complete_grad_vec);
      CpuOrtValueToVec(model_with_nominal_state->NamedParameters().at(name)->Data(), nominal_data_vec);
      CpuOrtValueToVec(model_with_nominal_state->NamedParameters().at(name)->Gradient(), nominal_grad_vec);
#endif

      for (size_t j = 0; j < complete_data_vec.size(); ++j) {
        ASSERT_EQ(complete_data_vec[j], nominal_data_vec[j]);
        ASSERT_EQ(complete_grad_vec[j], nominal_grad_vec[j]);
      }
    }

    std::vector<OrtValue> complete_eval_fetches;
    std::vector<OrtValue> nominal_eval_fetches;
    ASSERT_STATUS_OK(model_with_complete_state->EvalStep(inputs, complete_eval_fetches));
    ASSERT_STATUS_OK(model_with_nominal_state->EvalStep(inputs, nominal_eval_fetches));

    ASSERT_GT(complete_eval_fetches.size(), 0);
    for (size_t i = 0; i < complete_eval_fetches.size(); ++i) {
      ASSERT_TRUE(complete_eval_fetches[i].IsTensor());
      ASSERT_TRUE(nominal_eval_fetches[i].IsTensor());
      const Tensor& complete_tensor = complete_eval_fetches[i].Get<Tensor>();
      const Tensor& nominal_tensor = nominal_eval_fetches[i].Get<Tensor>();
      ASSERT_EQ(complete_tensor.Shape(), nominal_tensor.Shape());
      ASSERT_EQ(complete_tensor.DataType(), nominal_tensor.DataType());

      std::vector<float> complete_eval_fetches_vec;
      std::vector<float> nominal_eval_fetches_vec;
#if defined(USE_CUDA)
      CudaOrtValueToCpuVec(complete_eval_fetches[i], complete_eval_fetches_vec);
      CudaOrtValueToCpuVec(nominal_eval_fetches[i], nominal_eval_fetches_vec);
#else
      CpuOrtValueToVec(complete_eval_fetches[i], complete_eval_fetches_vec);
      CpuOrtValueToVec(nominal_eval_fetches[i], nominal_eval_fetches_vec);
#endif

      for (size_t j = 0; j < complete_eval_fetches_vec.size(); ++j) {
        ASSERT_EQ(complete_eval_fetches_vec[j], nominal_eval_fetches_vec[j]);
      }
    }
  }
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
