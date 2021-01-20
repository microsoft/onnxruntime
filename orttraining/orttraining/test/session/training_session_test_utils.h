// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "gtest/gtest.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "core/common/path_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/test/optimizer/horizontal_parallel_test_utils.h"

#include "orttraining/training_ops/cpu/controlflow/event_pool.h"  // TODO: move with PipelineBatchPlanner

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

using namespace onnxruntime::training;

namespace onnxruntime {
namespace test {
namespace training_session_test_utils {

constexpr auto ORIGINAL_MODEL_PATH = ORT_TSTR("testdata/test_training_model.onnx");
constexpr auto BACKWARD_MODEL_PATH = ORT_TSTR("testdata/temp_backward_model.onnx");
constexpr const char* const k_adam_optimizer_op_name = "AdamOptimizer";
constexpr const char* const k_lamb_optimizer_op_name = "LambOptimizer";
const std::vector<std::string> WEIGHT_NAMES = {"W1", "W2", "W3", "B1", "B2", "B3"};
const std::unordered_map<std::string, std::vector<int64_t>> WEIGHT_TO_SHAPE_MAP = {
    {"B3", {10}},
    {"W1", {784, 128}},
    {"W2", {128, 32}},
    {"B2", {32}},
    {"W3", {32, 10}},
    {"B1", {128}}};

void GenerateOptimizerConfig(const std::string optimizer_name,
                             const bool use_mixed_precision_moments,
                             training::TrainingSession::TrainingConfiguration& config);

template <class T>
void GenerateOptimizerInitialState(const std::string& optimizer_op_name, 
                                    const T init_moment_value, 
                                    training::TrainingSession::OptimizerState& optimizer_state);

void SeparateStateTensors(const NameMLValMap& training_state, 
                          NameMLValMap& model_state, 
                          training::TrainingSession::OptimizerState& optimizer_state);

void VerifyState(const DataTransferManager& data_transfer_mgr, const NameMLValMap& expected_state, const NameMLValMap& actual_state);

void VerifyOptimizerState(const DataTransferManager& data_transfer_manager, 
                          const training::TrainingSession::OptimizerState& expected_state, 
                          const training::TrainingSession::OptimizerState& actual_state);

std::unordered_set<std::string> GetModelOutputNames(const InferenceSession& session);

training::TrainingSession::TrainingConfiguration MakeBasicTrainingConfig();

/**
 * Run a training session for this model for 1 step, using batch size of 1 and synthetic input data.
 * @param so - SessionOptions for this run.
 * @param forward_model_file - Model file to be run.
 * @param config - Training session config
 * @return TrainingSession for this run.
 */
std::unique_ptr<training::TrainingSession> BuildAndRunTrainingSessionWithChecks(
    const SessionOptions& so, const PathString& forward_model_file,
    const training::TrainingSession::TrainingConfiguration& config);


// DistributedRunTestContext provides a method to override existing DistributedRunTestContext instance.
// This is for test purpose only. Please don't use it for other scenarios.
class DistributedRunTestContext : public DistributedRunContext
{
public:
    DistributedRunTestContext(const TrainingSession::TrainingConfiguration &config)
        : DistributedRunContext(config.distributed_config.world_rank,
                                config.distributed_config.world_size,
                                config.distributed_config.local_rank,
                                config.distributed_config.local_size,
                                config.distributed_config.data_parallel_size,
                                config.distributed_config.horizontal_parallel_size,
                                config.distributed_config.pipeline_parallel_size)
    {
    }

    // Reset the static DistributedRunContext object with new value.
    void ResetDistributedRunContext(){
      DistributedRunContext::GetRunConfig() = params_;
      auto& dp_group = DistributedRunContext::GetWorkerGroup(WorkerGroupType::DataParallel);
      dp_group = groups_[WorkerGroupType::DataParallel];

      auto& hp_group = DistributedRunContext::GetWorkerGroup(WorkerGroupType::HorizontalParallel);
      hp_group = groups_[WorkerGroupType::HorizontalParallel];

      auto& mp_group = DistributedRunContext::GetInstance().GetWorkerGroup(WorkerGroupType::PipelineParallel);
      mp_group = groups_[WorkerGroupType::PipelineParallel];
    }
};

}  // namespace training_session_test_utils
}  // namespace test
}  // namespace onnxruntime
