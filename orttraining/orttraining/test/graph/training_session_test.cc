// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <thread>

#include "gtest/gtest.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "core/common/path_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/test/graph/training_session_test_utils.h"

#include "orttraining/training_ops/cpu/controlflow/event_pool.h"  // TODO: move with PipelineBatchPlanner

#ifdef USE_CUDA
#include "bert_toy_fetches.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace google::protobuf::util;
using namespace onnxruntime::path_utils;

namespace onnxruntime {
namespace test {

static void RunTrainingSessionLoadOptimTests(std::string optim_name) {
  auto config = MakeBasicTrainingConfig();
  GenerateOptimizerConfig(optim_name, false, config);

  TrainingSession::OptimizerState init_optimizer_state{};
  GenerateOpimizerInitialState(optim_name, init_optimizer_state);

  config.init_optimizer_states = init_optimizer_state;
  SessionOptions so{};
  std::unique_ptr<TrainingSession> training_session = BuildAndRunTrainingSessionWithChecks(so, ORIGINAL_MODEL_PATH, config);

  NameMLValMap training_state{};
  ORT_ENFORCE(training_session->GetStateTensors(training_state).IsOK());
  const auto& data_transfer_manager = training_session->GetDataTransferManager();

  NameMLValMap model_state{};
  TrainingSession::OptimizerState actual_optimizer_state{};
  SeparateStateTensors(training_state, model_state, actual_optimizer_state);
  VerifyOptimizerState(data_transfer_manager, init_optimizer_state, actual_optimizer_state);
}

TEST(TrainingSessionTest, LoadOptimState_Adam) {
  RunTrainingSessionLoadOptimTests(k_adam_optimizer_op_name);
}

#ifdef USE_CUDA
// LambOptimizer op is registered for Cuda EP only
TEST(TrainingSessionTest, LoadOptimState_Lamb) {
  RunTrainingSessionLoadOptimTests(k_lamb_optimizer_op_name);
}
#endif

}  // namespace test
}  // namespace onnxruntime
