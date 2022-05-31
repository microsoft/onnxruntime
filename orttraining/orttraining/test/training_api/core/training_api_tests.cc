// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)
#include <thread>
#include <random>

#include "gtest/gtest.h"
#include "test/framework/test_utils.h"
#include "core/common/path_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "orttraining/training_api/include/utils.h"

#include "orttraining/training_api/include/interfaces.h"

using namespace onnxruntime::training;
using namespace onnxruntime::training::api;
using namespace onnxruntime::training::api::utils;
using namespace onnxruntime::path_utils;

namespace onnxruntime {
namespace training {

namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

template <typename T>
void OrtValueToVec(OrtValue& val, std::vector<T>& output) {
  const Tensor& tensor = val.Get<Tensor>();
  int64_t num_elem = tensor.Shape().Size();
  const T* val_ptr = tensor.template Data<T>();
  output.assign(val_ptr, val_ptr + num_elem);
}

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
  CreateInputOrtValue<float>(dims, data, &input);
}

TEST(TrainingApiTest, ModuleTrainStep) {
  auto model_uri = MODEL_FOLDER "gradient_graph.onnx";

  CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ORT_ENFORCE(LoadCheckpoint(checkpoint_to_load_path, state).IsOK());

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));
  auto module_sess = std::make_unique<onnxruntime::InferenceSession>(session_option, *env);
  ORT_THROW_IF_ERROR(module_sess->Load(model_uri));
  ORT_THROW_IF_ERROR(module_sess->Initialize());
  auto model = std::make_unique<Module>(state.module_checkpoint_state.named_parameters, module_sess.get());

  OrtValue input, target;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
  CreateInputOrtValue<int32_t>(std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1), &target);
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
    ORT_ENFORCE(model->TrainStep(inputs, fetches).IsOK());
    bias_grad = bias_param->Gradient();

    if (step > 1) {
      OrtValueToVec(bias_grad, current_bias_grad_vec);
      for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
        ORT_ENFORCE(current_bias_grad_vec[i] == single_bias_grad_vec[i] * step);
      }
    } else {
      OrtValueToVec(bias_grad, single_bias_grad_vec);
    }
  }
  // reset grad
  ORT_ENFORCE(model->ResetGrad().IsOK());

  // run a single step
  std::vector<OrtValue>& inputs = *data_loader.begin();
  std::vector<OrtValue> fetches;
  ORT_ENFORCE(model->TrainStep(inputs, fetches).IsOK());
  OrtValueToVec(bias_grad, current_bias_grad_vec);
  for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
    ORT_ENFORCE(current_bias_grad_vec[i] == single_bias_grad_vec[i]);
  }
}

TEST(TrainingApiTest, OptimStep) {
  auto model_uri = MODEL_FOLDER "gradient_graph.onnx";
  auto optim_uri = MODEL_FOLDER "adamw.onnx";

  CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ORT_ENFORCE(LoadCheckpoint(checkpoint_to_load_path, state).IsOK());

  onnxruntime::SessionOptions session_option;
  std::unique_ptr<Environment> env;
  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));
  auto module_sess = std::make_unique<onnxruntime::InferenceSession>(session_option, *env);
  auto optim_sess = std::make_unique<onnxruntime::InferenceSession>(session_option, *env);
  ORT_THROW_IF_ERROR(module_sess->Load(model_uri));
  ORT_THROW_IF_ERROR(module_sess->Initialize());
  ORT_THROW_IF_ERROR(optim_sess->Load(optim_uri));
  ORT_THROW_IF_ERROR(optim_sess->Initialize());

  auto model = std::make_unique<Module>(state.module_checkpoint_state.named_parameters, module_sess.get());
  auto optim = std::make_unique<Optimizer>(state.module_checkpoint_state.named_parameters, optim_sess.get());

  OrtValue input, target;
  GenerateRandomInput(std::array<int64_t, 2>{2, 784}, input);
  CreateInputOrtValue<int32_t>(std::array<int64_t, 1>{2}, std::vector<int32_t>(2, 1), &target);
  auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

  size_t step = 0;
  std::string param_name = "fc2.weight";

  // before training, check if optim state is initialized to 0
  OptimizerCheckpointState optimizer_states;
  ORT_ENFORCE(optim->GetStateDict(optimizer_states).IsOK());
  ParameterOptimizerState& param_state = optimizer_states.group_named_optimizer_states["group0"]->param_named_optimizer_states.at(param_name);
  OrtValue& moment_1 = param_state.momentum_named_states.at("momentum0");

  std::vector<float> moment_1_vec;
  OrtValueToVec(moment_1, moment_1_vec);
  for (size_t i = 0; i < moment_1_vec.size(); i++) {
    ORT_ENFORCE(moment_1_vec[i] == 0.0f);
  }

  for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    step += 1;
    std::vector<OrtValue>& inputs = *it;
    std::vector<OrtValue> fetches;
    ORT_ENFORCE(model->TrainStep(inputs, fetches).IsOK());
    ORT_ENFORCE(optim->Step().IsOK());

    // get optim state and check if it is updated
    OrtValueToVec(moment_1, moment_1_vec);
    for (size_t i = 0; i < moment_1_vec.size(); i++) {
      if (moment_1_vec[i] != 0.0f) {
        // moment was updated
        break;
      }
    }
  }
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
#endif
