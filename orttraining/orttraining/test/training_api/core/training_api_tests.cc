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
#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif

using json = nlohmann::json;
using namespace onnxruntime::training::api;

namespace onnxruntime {
namespace training {
namespace test {

namespace {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

constexpr int64_t total_step_count = 100;
constexpr float initial_lr = 1e-3f;
constexpr int64_t resume_step = total_step_count / 2;

void CompareValue(float expected, float output, float rtol = 1e-4, float atol = 1e-5) {
  ASSERT_NEAR(expected, output, atol);
  ASSERT_NEAR(expected, output, rtol * std::abs(expected));
}

/**
 * @brief Prepare the model and optimizer before launching the test function.
 * Be noted: though the state might not be used by test_function, we still need to pass it to
 * make sure `state` lives through until test_function is finished.
 *
 * @param run_cuda If true, running on cuda; otherwise, running on cpu.
 * @param need_eval If true, load eval model when building Module.
 * @param need_opt If true, don't build Optimizer, only build Module.
 */
void PrepareModelAndOptimizerForTest(bool run_cuda,
                                     bool need_eval,
                                     bool need_opt,
                                     std::function<void(std::shared_ptr<Module>,
                                                        std::shared_ptr<Optimizer>,
                                                        CheckpointState&,
                                                        bool)>
                                         test_function) {
  std::vector<std::shared_ptr<IExecutionProvider>> providers;
  if (run_cuda) {
    providers = {onnxruntime::test::DefaultCudaExecutionProvider()};
  } else {
    providers = {onnxruntime::test::DefaultCpuExecutionProvider()};
  }

  auto model_uri = MODEL_FOLDER "training_model.onnx";
  auto eval_model_uri = MODEL_FOLDER "eval_model.onnx";
  auto optim_uri = MODEL_FOLDER "adamw.onnx";

  CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ASSERT_STATUS_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_to_load_path, state));

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;

  ASSERT_STATUS_OK(Environment::Create(nullptr, env));

  std::shared_ptr<Module> model;
  if (need_eval) {
    model = std::make_shared<Module>(
        ToUTF8String(model_uri), &state, onnxruntime::SessionOptions(),
        *env, providers, ToUTF8String(eval_model_uri));
  } else {
    model = std::make_shared<Module>(
        ToUTF8String(model_uri), &state, session_option,
        *env, providers);
  }

  std::shared_ptr<Optimizer> optim;
  if (need_opt) {
    optim = std::make_shared<Optimizer>(
        ToUTF8String(optim_uri), &state, session_option,
        *env, providers);
  }

  test_function(model, optim, state, run_cuda);
}

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
    const std::vector<std::string>& momentum_keys,
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
        cur_param_optimizer_states.momentum_named_states.insert({state_name, std::move(param_moment_state)});
      }
    }
  }

  return Status::OK();
}

}  // namespace

TEST(TrainingApiTest, ModuleParametersSize) {
  auto run_test = [](std::shared_ptr<Module> model,
                     std::shared_ptr<Optimizer> /*optim*/,
                     CheckpointState& /*state*/,
                     bool /*run_cuda*/)
      -> void {
    size_t params_size = 0;
    for (auto& param : model->Parameters()) {
      params_size += param->Data().Get<Tensor>().Shape().Size();
    }

    // ((500*784) + 500 + (10*500) + 10) = 397510
    ASSERT_EQ(params_size, 397510);
    ASSERT_EQ(model->GetParametersSize(), 397510);
  };

  PrepareModelAndOptimizerForTest(false /*run_cuda*/, false /*need_eval*/, false /*need_opt*/, run_test);
#ifdef USE_CUDA
  PrepareModelAndOptimizerForTest(true /*run_cuda*/, false /*need_eval*/, false /*need_opt*/, run_test);
#endif
}

TEST(TrainingApiTest, ModuleCopyBufferToParameters) {
  auto run_test = [](std::shared_ptr<Module> model,
                     std::shared_ptr<Optimizer> /*optim*/,
                     CheckpointState& /*state*/,
                     bool /*run_cuda*/)
      -> void {
    int64_t params_size = static_cast<int64_t>(model->GetParametersSize());
    std::vector<float> expected_param_buffer(params_size);
    GenerateRandomData(expected_param_buffer);

    OrtValue input_params;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),
                         {params_size},
                         reinterpret_cast<void*>(expected_param_buffer.data()),
                         onnxruntime::test::TestCPUExecutionProvider()->GetAllocator(OrtMemTypeDefault)->Info(),
                         input_params, 0);
    ASSERT_STATUS_OK(model->CopyBufferToParameters(input_params));

    OrtValue output_params;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), {params_size},
                         onnxruntime::test::TestCPUExecutionProvider()->GetAllocator(OrtMemTypeDefault),
                         output_params);
    ASSERT_STATUS_OK(model->CopyParametersToBuffer(output_params));

    const float* buffer = output_params.Get<Tensor>().Data<float>();
    ASSERT_TRUE(nullptr != buffer);
    for (int64_t i = 0; i < params_size; i++) {
      ASSERT_TRUE(*(buffer + i) == expected_param_buffer[i]);
    }
  };

  PrepareModelAndOptimizerForTest(false /*run_cuda*/, false /*need_eval*/, false /*need_opt*/, run_test);
#ifdef USE_CUDA
  PrepareModelAndOptimizerForTest(true /*run_cuda*/, false /*need_eval*/, false /*need_opt*/, run_test);
#endif
}

TEST(TrainingApiTest, ModuleTrainStep) {
  auto run_test = [](std::shared_ptr<Module> model, std::shared_ptr<Optimizer> /*optim*/,
                     CheckpointState& /*state*/,
                     bool run_cuda)
      -> void {
    ASSERT_EQ(model->GetTrainingModelOutputCount(), 1);
    OrtValue input, target;
    GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
    onnxruntime::test::CreateInputOrtValueOnCPU<int32_t>(
        std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1), &target);
    auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

    size_t step = 0;
    std::vector<float> single_bias_grad_vec, current_bias_grad_vec;
    std::string param_name = "fc2.weight";
    std::shared_ptr<Parameter> bias_param = model->NamedParameters()[param_name];
    OrtValue& bias_grad = bias_param->Gradient();

    for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
      step += 1;
      std::vector<OrtValue>& inputs = *it;
      std::vector<OrtValue> fetches;
      ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
      ASSERT_EQ(fetches.size(), 1U);
      bias_grad = bias_param->Gradient();

      if (step > 1) {
        if (run_cuda) {
          CudaOrtValueToCpuVec(bias_grad, current_bias_grad_vec);
        } else {
          CpuOrtValueToVec(bias_grad, current_bias_grad_vec);
        }
        for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
          ASSERT_EQ(current_bias_grad_vec[i], single_bias_grad_vec[i] * step);
        }
      } else {
        if (run_cuda) {
          CudaOrtValueToCpuVec(bias_grad, single_bias_grad_vec);
        } else {
          CpuOrtValueToVec(bias_grad, single_bias_grad_vec);
        }
      }
    }
    // reset grad
    ASSERT_STATUS_OK(model->LazyResetGrad());

    // run a single step
    std::vector<OrtValue>& inputs = *data_loader.begin();
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
    if (run_cuda) {
      CudaOrtValueToCpuVec(bias_grad, current_bias_grad_vec);
    } else {
      CpuOrtValueToVec(bias_grad, current_bias_grad_vec);
    }
    for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
      ASSERT_EQ(current_bias_grad_vec[i], single_bias_grad_vec[i]);
    }
  };

  PrepareModelAndOptimizerForTest(false /*run_cuda*/, false /*need_eval*/, false /*need_opt*/, run_test);
#ifdef USE_CUDA
  PrepareModelAndOptimizerForTest(true /*run_cuda*/, false /*need_eval*/, false /*need_opt*/, run_test);
#endif
}

TEST(TrainingApiTest, OptimizerCreatedWithoutOptimizerCheckpointState) {
  auto run_test = [](std::shared_ptr<Module> model, std::shared_ptr<Optimizer> optim,
                     CheckpointState& /*state*/,
                     bool run_cuda)
      -> void {
    {
      // Check if optimizer state is initialized to 0.
      OptimizerCheckpointState optimizer_states;
      ASSERT_STATUS_OK(optim->GetStateDict(optimizer_states));

      for (auto& p : model->NamedParameters()) {
        auto param_name = p.first;
        ParameterOptimizerState& param_state =
            optimizer_states.group_named_optimizer_states["group0"]->param_named_optimizer_states.at(param_name);
        for (auto& param_p : param_state.momentum_named_states) {
          std::vector<float> moment_vec;
          if (run_cuda)
            CudaOrtValueToCpuVec(param_state.momentum_named_states.at(param_p.first), moment_vec);
          else
            CpuOrtValueToVec(param_state.momentum_named_states.at(param_p.first), moment_vec);
          for (size_t i = 0; i < moment_vec.size(); i++) {
            ASSERT_EQ(moment_vec[i], 0.0f);
          }
        }
      }
    }
  };

  PrepareModelAndOptimizerForTest(false /*run_cuda*/, false /*need_eval*/, true /*need_opt*/, run_test);
#ifdef USE_CUDA
  PrepareModelAndOptimizerForTest(true /*run_cuda*/, false /*need_eval*/, true /*need_opt*/, run_test);
#endif
}

TEST(TrainingApiTest, OptimizerCreatedWithOptimizerCheckpointState) {
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

    std::shared_ptr<Module> model = std::make_shared<Module>(
        ToUTF8String(model_uri), &state, session_option,
        *env, providers);

    // Load state dict from faked optimizer checkpoint state.
    CheckpointState new_state = state;
    OptimizerCheckpointState& external_optimizer_checkpoint_state = new_state.optimizer_checkpoint_state;
    ASSERT_STATUS_OK(CreateFakeOptimizerCheckpointStateOnCPU(model->NamedParameters(),
                                                             {"momentum0", "momentum1"},
                                                             external_optimizer_checkpoint_state));
    std::shared_ptr<Optimizer> optim = std::make_shared<Optimizer>(
        ToUTF8String(optim_uri), &new_state, session_option, *env, providers);

    // After loading state dict, check if optim state is updated to new states.
    OptimizerCheckpointState optimizer_states;
    ASSERT_STATUS_OK(optim->GetStateDict(optimizer_states));

    for (auto& p : model->NamedParameters()) {
      auto param_name = p.first;
      ParameterOptimizerState& param_state =
          optimizer_states.group_named_optimizer_states["group0"]->param_named_optimizer_states.at(param_name);

      ParameterOptimizerState& external_param_state =
          external_optimizer_checkpoint_state.group_named_optimizer_states["group0"]
              ->param_named_optimizer_states.at(param_name);
      for (auto& param_p : param_state.momentum_named_states) {
        std::vector<float> moment_vec;
        if (run_cuda) {
          CudaOrtValueToCpuVec(param_state.momentum_named_states.at(param_p.first), moment_vec);
        } else {
          CpuOrtValueToVec(param_state.momentum_named_states.at(param_p.first), moment_vec);
        }
        std::vector<float> external_moment_vect;

        if (run_cuda) {
          CudaOrtValueToCpuVec(external_param_state.momentum_named_states.at(param_p.first), external_moment_vect);
        } else {
          CpuOrtValueToVec(external_param_state.momentum_named_states.at(param_p.first), external_moment_vect);
        }

        ASSERT_EQ(moment_vec.size(), external_moment_vect.size());
        for (size_t i = 0; i < moment_vec.size(); i++) {
          ASSERT_EQ(moment_vec[i], external_moment_vect[i]);
        }
      }
    }
  }
}

TEST(TrainingApiTest, OptimizerRestoreFromCheckpointState) {
  auto run_test = [](std::shared_ptr<Module> model, std::shared_ptr<Optimizer> optim,
                     CheckpointState& /*state*/,
                     bool run_cuda)
      -> void {
    // Load state dict from faked optimizer checkpoint state.
    OptimizerCheckpointState external_optimizer_checkpoint_state;
    ASSERT_STATUS_OK(CreateFakeOptimizerCheckpointStateOnCPU(model->NamedParameters(), {"momentum0", "momentum1"},
                                                             external_optimizer_checkpoint_state));
    ASSERT_STATUS_OK(optim->LoadStateDict(external_optimizer_checkpoint_state));

    // After loading state dict, validate optim state is updated to new states.
    OptimizerCheckpointState optimizer_states;
    ASSERT_STATUS_OK(optim->GetStateDict(optimizer_states));

    for (auto& p : model->NamedParameters()) {
      auto param_name = p.first;
      ParameterOptimizerState& param_state =
          optimizer_states.group_named_optimizer_states["group0"]->param_named_optimizer_states.at(param_name);

      ParameterOptimizerState& external_param_state =
          external_optimizer_checkpoint_state.group_named_optimizer_states["group0"]
              ->param_named_optimizer_states.at(param_name);
      for (auto& param_p : param_state.momentum_named_states) {
        std::vector<float> moment_vec;
        if (run_cuda) {
          CudaOrtValueToCpuVec(param_state.momentum_named_states.at(param_p.first), moment_vec);
        } else {
          CpuOrtValueToVec(param_state.momentum_named_states.at(param_p.first), moment_vec);
        }

        std::vector<float> external_moment_vect;
        if (run_cuda) {
          CudaOrtValueToCpuVec(external_param_state.momentum_named_states.at(param_p.first), external_moment_vect);
        } else {
          CpuOrtValueToVec(external_param_state.momentum_named_states.at(param_p.first), external_moment_vect);
        }

        ASSERT_EQ(moment_vec.size(), external_moment_vect.size());
        for (size_t i = 0; i < moment_vec.size(); i++) {
          ASSERT_EQ(moment_vec[i], external_moment_vect[i]);
        }
      }
    }
  };

  PrepareModelAndOptimizerForTest(false /*run_cuda*/, false /*need_eval*/, true /*need_opt*/, run_test);
#ifdef USE_CUDA
  PrepareModelAndOptimizerForTest(true /*run_cuda*/, false /*need_eval*/, true /*need_opt*/, run_test);
#endif
}

TEST(TrainingApiTest, OptimizerStep) {
  auto run_test = [](std::shared_ptr<Module> model, std::shared_ptr<Optimizer> optim,
                     CheckpointState& /*state*/,
                     bool run_cuda)
      -> void {
    OrtValue input, target;
    GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
    onnxruntime::test::CreateInputOrtValueOnCPU<int32_t>(
        std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1), &target);
    auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

    size_t step = 0;
    std::string param_name = "fc2.weight";

    // Before training, check if optim state is initialized to 0
    std::vector<float> param_vec_before_optimizer_step;
    {
      OptimizerCheckpointState optimizer_states;
      ASSERT_STATUS_OK(optim->GetStateDict(optimizer_states));
      ParameterOptimizerState& param_state =
          optimizer_states.group_named_optimizer_states["group0"]->param_named_optimizer_states.at(param_name);
      OrtValue& moment_1 = param_state.momentum_named_states.at("momentum0");
      std::vector<float> moment_1_vec;
      if (run_cuda) {
        CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Data(), param_vec_before_optimizer_step);
        CudaOrtValueToCpuVec(moment_1, moment_1_vec);
      } else {
        CpuOrtValueToVec(model->NamedParameters().at(param_name)->Data(), param_vec_before_optimizer_step);
        CpuOrtValueToVec(moment_1, moment_1_vec);
      }

      for (size_t i = 0; i < moment_1_vec.size(); i++) {
        ASSERT_EQ(moment_1_vec[i], 0.0f);
      }
    }

    for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
      step += 1;
      std::vector<OrtValue>& inputs = *it;
      std::vector<OrtValue> fetches;
      ASSERT_STATUS_OK(model->TrainStep(inputs, fetches));
      std::vector<float> grads;
      if (run_cuda) {
        CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Gradient(), grads);
      } else {
        CpuOrtValueToVec(model->NamedParameters().at(param_name)->Gradient(), grads);
      }

      ASSERT_STATUS_OK(optim->Step());

      // Get the optimizer state and check if it is updated
      {
        OptimizerCheckpointState optimizer_states;
        ASSERT_STATUS_OK(optim->GetStateDict(optimizer_states));
        ParameterOptimizerState& param_state =
            optimizer_states.group_named_optimizer_states["group0"]->param_named_optimizer_states.at(param_name);
        OrtValue& moment_1 = param_state.momentum_named_states.at("momentum0");
        std::vector<float> moment_1_vec;
        CudaOrtValueToCpuVec(moment_1, moment_1_vec);
        for (size_t i = 0; i < moment_1_vec.size(); i++) {
          if (grads[i] != 0.0f) {
            ASSERT_NE(moment_1_vec[i], 0.0f);
          }
        }

        std::vector<float> param_vec_after_optimizer_step;
        if (run_cuda) {
          CudaOrtValueToCpuVec(model->NamedParameters().at(param_name)->Data(), param_vec_after_optimizer_step);
        } else {
          CpuOrtValueToVec(model->NamedParameters().at(param_name)->Data(), param_vec_after_optimizer_step);
        }

        for (size_t i = 0; i < param_vec_after_optimizer_step.size(); ++i) {
          if (grads[i] != 0.0f && moment_1_vec[i] != 0.0f) {
            ASSERT_NE(param_vec_after_optimizer_step[i], param_vec_before_optimizer_step[i]);
          }
        }
      }
    }
  };

  PrepareModelAndOptimizerForTest(false /*run_cuda*/, false /*need_eval*/, true /*need_opt*/, run_test);
#ifdef USE_CUDA
  PrepareModelAndOptimizerForTest(true /*run_cuda*/, false /*need_eval*/, true /*need_opt*/, run_test);
#endif
}

TEST(TrainingApiTest, ModuleExportModelForInferencing) {
  auto eval_model_uri = MODEL_FOLDER "eval_model.onnx";
  ONNX_NAMESPACE::ModelProto eval_model;
  ORT_THROW_IF_ERROR(Model::Load(eval_model_uri, eval_model));
  std::unique_ptr<Environment> env;
  ASSERT_STATUS_OK(Environment::Create(nullptr, env));

  auto run_test =
      [&eval_model, &env](std::shared_ptr<Module> model, std::shared_ptr<Optimizer>, CheckpointState& /*state*/,
                          bool /*run_cuda*/)
      -> void {
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

    ONNX_NAMESPACE::ModelProto inference_model;
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
  };

  PrepareModelAndOptimizerForTest(false /*run_cuda*/, true /*need_eval*/, false /*need_opt*/, run_test);
#ifdef USE_CUDA
  PrepareModelAndOptimizerForTest(true /*run_cuda*/, true /*need_eval*/, false /*need_opt*/, run_test);
#endif
}

void TestLRSchduler(const std::basic_string<ORTCHAR_T>& test_file_name, float initial_lr, int64_t total_step_count,
                    int64_t warmup_step_count) {
  auto run_test =
      [&test_file_name, initial_lr, total_step_count, warmup_step_count](
          std::shared_ptr<Module> model, std::shared_ptr<Optimizer> optim, CheckpointState& /*state*/,
          bool /*run_cuda*/)
      -> void {
    OrtValue input, target;
    GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
    onnxruntime::test::CreateInputOrtValueOnCPU<int32_t>(
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
      OptimizerCheckpointState optimizer_checkpoint_states;
      ASSERT_STATUS_OK(CreateFakeOptimizerCheckpointStateOnCPU(model->NamedParameters(), {"momentum0", "momentum1"},
                                                               optimizer_checkpoint_states));
      auto group_opt_state = optimizer_checkpoint_states.group_named_optimizer_states["group0"];
      /// Reset optimizer states to match the initial state we want to test.
      group_opt_state->step = resume_step;
      group_opt_state->initial_lr = initial_lr;
      ASSERT_STATUS_OK(optim->LoadStateDict(optimizer_checkpoint_states));
    }

    // KNOWN ISSUE: LinearLRScheduler by default use optim's states to calculate the first step's learning rate.
    // If we restored it after creation, it will only affect the learning rate from the second step.
    auto scheduler = std::make_unique<LinearLRScheduler>(
        optim, warmup_step_count, total_step_count);

    for (auto it = test_data.begin(); it != test_data.end(); ++it) {
      OptimizerCheckpointState optimizer_states;
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
  };

  PrepareModelAndOptimizerForTest(false /*run_cuda*/, false /*need_eval*/, true /*need_opt*/, run_test);
#ifdef USE_CUDA
  PrepareModelAndOptimizerForTest(true /*run_cuda*/, false /*need_eval*/, true /*need_opt*/, run_test);
#endif
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

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
