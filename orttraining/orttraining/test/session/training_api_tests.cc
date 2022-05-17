// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)
#include <thread>

#include "gtest/gtest.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
// #include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "core/common/path_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/models/runner/training_runner.h"
#include "core/framework/tensorprotoutils.h"

#include "orttraining/training_api/include/utils.h"
#include "orttraining/training_api/include/interfaces.h"

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace onnxruntime::training::api;
using namespace google::protobuf::util;
using namespace onnxruntime::path_utils;

#ifdef USE_CUDA
namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name);

}  // namespace onnxruntime
#endif

namespace onnxruntime {
namespace training {

namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

template <typename T>
static void OrtValueToVec(OrtValue& val, std::vector<T>& output) {
  const Tensor& tensor = val.Get<Tensor>();
  int64_t num_ele = tensor.Shape().Size();
  const float* val_ptr = tensor.template Data<float>();
  output.assign(val_ptr, val_ptr + num_ele);
}

TEST(TrainingApiTest, ModuleTrainStep) {
  auto model_uri = MODEL_FOLDER "gradient_graph.onnx";

  CheckpointState state;
  auto checkpoint_to_load_path = MODEL_FOLDER "checkpoint.ckpt";
  ORT_ENFORCE(LoadCheckpoint(checkpoint_to_load_path, state).IsOK());

  auto module_sess = std::make_unique<Module>(model_uri, state.module_checkpoint_state.named_parameters);

  OrtValue input, target;
  // hard coded each sample to have 4 elements so far.
  // todo: we can make it support more generic once we are clear what our offline process graph needed.
  CreateInputOrtValue({2, 784}, std::vector<float_t>(1568, 1), &input);
  CreateInputOrtValue({2}, std::vector<int32_t>(2, 1), &target);
  auto data_loader = std::vector<std::vector<OrtValue>>(4, std::vector<OrtValue>{input, target});

  size_t step = 0;
  std::vector<float> before_train_vec, single_bias_grad_vec, current_bias_grad_vec, single_grad_vec, accumulated_grad_vec;
  std::string param_name = "fc2.weight";
  std::shared_ptr<Parameter> bias_param = module_sess->named_parameters()[param_name];
  OrtValue& bias_grad = bias_param->Gradient();
  OrtValueToVec(bias_grad, before_train_vec);

  for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    step += 1;
    std::vector<OrtValue>& inputs = *it;
    std::vector<OrtValue> fetches;
    ORT_ENFORCE(module_sess->TrainStep(inputs, fetches).IsOK());
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
  ORT_ENFORCE(module_sess->ResetGrad().IsOK());

  // run a single step
  std::vector<OrtValue>& inputs = *data_loader.begin();
  std::vector<OrtValue> fetches;
  ORT_ENFORCE(module_sess->TrainStep(inputs, fetches).IsOK());
  OrtValueToVec(bias_grad, current_bias_grad_vec);
  for (size_t i = 0; i < current_bias_grad_vec.size(); i++) {
    ORT_ENFORCE(current_bias_grad_vec[i] == single_bias_grad_vec[i]);
  }
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
#endif