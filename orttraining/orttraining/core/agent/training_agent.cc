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

  auto bp = inference_session_.GetBreakpointAndEndPoint();
  fw_program_counter_end_ = bp.first - 1;
  bw_program_counter_end_ = bp.second;
}

TrainingAgent::~TrainingAgent(){};

std::vector<OrtValue> TrainingAgent::RunForward(std::vector<OrtValue>& feeds, PartialGraphExecutionState& state) {
  state.SetProgramCounterStart(0);
  state.SetProgramCounterEnd(fw_program_counter_end_);
  return RunCore(feeds, state, *fw_feeds_fetches_manager_);
}

std::vector<OrtValue> TrainingAgent::RunBackward(std::vector<OrtValue>& feeds, PartialGraphExecutionState& state) {
  state.SetProgramCounterStart(fw_program_counter_end_ + 1);
  state.SetProgramCounterEnd(bw_program_counter_end_);
  return RunCore(feeds, state, *bw_feeds_fetches_manager_);
}

std::vector<OrtValue> TrainingAgent::RunCore(std::vector<OrtValue>& feeds, PartialGraphExecutionState& state, FeedsFetchesManager& feeds_fetches_manager) {
  auto fetches_size = feeds_fetches_manager.GetFeedsFetchesInfo().output_names.size();
  std::vector<OrtValue> fetches;
  fetches.resize(fetches_size);
  for (size_t index = 0; index < fetches_size; index += 1) {
    fetches[index] = {};
  }

  RunOptions run_options;
  auto status = inference_session_.Run(run_options, feeds, fetches, state, feeds_fetches_manager);
  if (!status.IsOK()) {
    throw std::runtime_error("Error in execution: " + status.ErrorMessage());
  }

  return fetches;
}

}  // namespace training
}  // namespace onnxruntime
