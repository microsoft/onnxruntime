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
#include "orttraining/test/session/training_session_test_utils.h"

#include "orttraining/training_ops/cpu/controlflow/event_pool.h"  // TODO: move with PipelineBatchPlanner

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace onnxruntime::path_utils;
using namespace onnxruntime::test::training_session_test_utils;

namespace onnxruntime {
namespace test {

static void RunTrainingSessionLoadOptimTests(std::string optim_name, bool mixed_precision, bool mixed_precision_moments) {
  auto config = MakeBasicTrainingConfig();
  if (mixed_precision) {
    TrainingSession::TrainingConfiguration::MixedPrecisionConfiguration mp{};
    mp.use_mixed_precision_initializers = true;
    config.mixed_precision_config = mp;
  }
  GenerateOptimizerConfig(optim_name, mixed_precision_moments, config);

  TrainingSession::OptimizerState init_optimizer_state{};
  if (mixed_precision_moments) {
    GenerateOptimizerInitialState<MLFloat16>(optim_name, MLFloat16(2.5f), init_optimizer_state);
  } else {
    GenerateOptimizerInitialState<float>(optim_name, 2.5f, init_optimizer_state);
  }

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

TEST(TrainingSessionTest, LoadOptimState_FullPrecision_FP32Moments_Adam) {
  RunTrainingSessionLoadOptimTests(k_adam_optimizer_op_name, false, false);
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(TrainingSessionTest, LoadOptimState_MixedPrecision_FP32Moments_Adam) {
  RunTrainingSessionLoadOptimTests(k_adam_optimizer_op_name, true, false);
}

TEST(TrainingSessionTest, LoadOptimState_MixedPrecision_FP16Moments_Adam) {
  RunTrainingSessionLoadOptimTests(k_adam_optimizer_op_name, true, true);
}

// LambOptimizer op is registered for Cuda EP only
TEST(TrainingSessionTest, LoadOptimState_FullPrecision_FP32Moments_Lamb) {
  RunTrainingSessionLoadOptimTests(k_lamb_optimizer_op_name, false, false);
}

// FP16 moments are not supported by Lamb
TEST(TrainingSessionTest, LoadOptimState_MixedPrecision_FP32Moments_Lamb) {
  RunTrainingSessionLoadOptimTests(k_lamb_optimizer_op_name, true, false);
}
#endif

}  // namespace test
}  // namespace onnxruntime
