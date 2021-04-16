// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/agent/training_agent.h"
#include "core/framework/utils.h"
#include "core/framework/feeds_fetches_manager.h"

namespace onnxruntime {
namespace training {

TrainingAgent::TrainingAgent(InferenceSession& session, const std::vector<std::string>& fw_feed_names,
                             const std::vector<std::string>& fw_fetches_names, const std::vector<OrtDevice>& fw_outputs_device_info,
                             const std::vector<std::string>& bw_feed_names, const std::vector<std::string>& bw_fetches_names,
                             const std::vector<OrtDevice>& bw_outputs_device_info) : inference_session_(session) {
  auto& session_state = session.GetSessionState();
  {
    FeedsFetchesManager::Create(fw_feed_names, fw_fetches_names, session_state.GetOrtValueNameIdxMap(), fw_feeds_fetches_manager_);
    auto& fetch_info = fw_feeds_fetches_manager_->GetMutableFetchesDeviceCopyInfo();
    for (size_t i = 0, end = fw_fetches_names.size(); i < end; ++i) {
      fetch_info[i].target_device = fw_outputs_device_info[i];
    }

    ORT_ENFORCE(utils::InitializeFeedFetchCopyInfo(session_state, *fw_feeds_fetches_manager_) == Status::OK());
  }
  {
    FeedsFetchesManager::Create(bw_feed_names, bw_fetches_names, session_state.GetOrtValueNameIdxMap(), bw_feeds_fetches_manager_);
    auto& fetch_info = bw_feeds_fetches_manager_->GetMutableFetchesDeviceCopyInfo();
    for (size_t i = 0, end = bw_fetches_names.size(); i < end; ++i) {
      fetch_info[i].target_device = bw_outputs_device_info[i];
    }

    ORT_ENFORCE(utils::InitializeFeedFetchCopyInfo(session_state, *bw_feeds_fetches_manager_) == Status::OK());
  }

  size_t end_point;
  size_t break_point = 0;
  const SequentialExecutionPlan& seq_exec_plan = *(session_state.GetExecutionPlan());
  const auto& exec_plan_vec = seq_exec_plan.execution_plan;
  end_point = exec_plan_vec.size() - 1;
  for (size_t program_counter = 0; program_counter < exec_plan_vec.size(); program_counter += 1) {
    const auto& node_exec_plan = exec_plan_vec[program_counter];
    auto node_index = node_exec_plan.node_index;
    if (session_state.GetKernel(node_index)->KernelDef().OpName() == "YieldOp") {
      break;
    }
    break_point += 1;
  }

  fw_program_counter_end_ = break_point - 1;
  bw_program_counter_end_ = end_point;
}

TrainingAgent::~TrainingAgent(){};

void TrainingAgent::RunForward(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches, PartialGraphExecutionState& state) {
  state.SetProgramCounterStart(0);
  state.SetProgramCounterEnd(fw_program_counter_end_);
  RunCore(feeds, fetches, state, *fw_feeds_fetches_manager_);
}

void TrainingAgent::RunBackward(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches, PartialGraphExecutionState& state) {
  state.SetProgramCounterStart(fw_program_counter_end_ + 1);
  state.SetProgramCounterEnd(bw_program_counter_end_);
  RunCore(feeds, fetches, state, *bw_feeds_fetches_manager_);
}

void TrainingAgent::RunCore(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches, PartialGraphExecutionState& state, FeedsFetchesManager& feeds_fetches_manager) {
  auto fetches_size = feeds_fetches_manager.GetFeedsFetchesInfo().output_names.size();
  fetches.resize(fetches_size);
  for (size_t index = 0; index < fetches_size; index += 1) {
    fetches[index] = {};
  }

  RunOptions run_options;
  auto status = inference_session_.PartialRun(run_options, feeds, fetches, state, feeds_fetches_manager);
  if (!status.IsOK()) {
    throw std::runtime_error("Error in execution: " + status.ErrorMessage());
  }
}

}  // namespace training
}  // namespace onnxruntime
