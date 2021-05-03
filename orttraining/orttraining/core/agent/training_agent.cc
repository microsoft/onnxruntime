// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/agent/training_agent.h"
#include "core/framework/utils.h"
#include "core/framework/feeds_fetches_manager.h"

namespace onnxruntime {
namespace training {

TrainingAgent::TrainingAgent(InferenceSession& session,
                             const std::vector<std::string>& fw_feed_names,
                             const std::vector<std::string>& fw_fetches_names,
                             const std::vector<OrtDevice>& fw_outputs_device_info,
                             const std::vector<std::string>& bw_feed_names,
                             const std::vector<std::string>& bw_fetches_names,
                             const std::vector<OrtDevice>& bw_outputs_device_info) : inference_session_(session) {
  auto& session_state = session.GetSessionState();
  CreateAndInitializeFeedsFetchesManager(session_state, fw_feed_names, fw_fetches_names, fw_outputs_device_info,
                                         fw_feeds_fetches_manager_);

  CreateAndInitializeFeedsFetchesManager(session_state, bw_feed_names, bw_fetches_names, bw_outputs_device_info,
                                         bw_feeds_fetches_manager_);

  size_t break_point = 0;
  const SequentialExecutionPlan& seq_exec_plan = *(session_state.GetExecutionPlan());
  const auto& exec_plan_vec = seq_exec_plan.execution_plan;
  for (size_t program_counter = 0; program_counter < exec_plan_vec.size(); program_counter += 1) {
    const auto& node_exec_plan = exec_plan_vec[program_counter];
    auto node_index = node_exec_plan.node_index;
    if (session_state.GetKernel(node_index)->KernelDef().OpName() == "YieldOp") {
      break;
    }
    break_point += 1;
  }

  fw_program_counter_end_ = break_point;
  bw_program_counter_end_ = exec_plan_vec.size();
}

TrainingAgent::~TrainingAgent() = default;

common::Status TrainingAgent::RunForward(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                         PartialGraphExecutionState& state) {
  state.SetProgramCounterStart(0);
  state.SetProgramCounterEnd(fw_program_counter_end_);
  return RunCore(feeds, fetches, state, *fw_feeds_fetches_manager_);
}

common::Status TrainingAgent::RunBackward(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                          PartialGraphExecutionState& state) {
  state.SetProgramCounterStart(fw_program_counter_end_);
  state.SetProgramCounterEnd(bw_program_counter_end_);
  state.SetReleaseOutputs(true);
  return RunCore(feeds, fetches, state, *bw_feeds_fetches_manager_);
}

common::Status TrainingAgent::RunCore(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                      PartialGraphExecutionState& state, FeedsFetchesManager& feeds_fetches_manager) {
  auto fetches_size = feeds_fetches_manager.GetFeedsFetchesInfo().output_names.size();
  fetches.resize(fetches_size, {});
  RunOptions run_options;
  return inference_session_.PartialRun(run_options, feeds, fetches, state, feeds_fetches_manager);
}

void TrainingAgent::CreateAndInitializeFeedsFetchesManager(const SessionState& session_state,
                                                           const std::vector<std::string>& feed_names,
                                                           const std::vector<std::string>& fetches_names,
                                                           const std::vector<OrtDevice>& outputs_device_info,
                                                           std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager) {
  FeedsFetchesManager::Create(feed_names, fetches_names, session_state.GetOrtValueNameIdxMap(), feeds_fetches_manager);
  auto& fetch_info = feeds_fetches_manager->GetMutableFetchesDeviceCopyInfo();
  for (size_t i = 0, end = fetches_names.size(); i < end; ++i) {
    fetch_info[i].target_device = outputs_device_info[i];
  }

  ORT_ENFORCE(utils::InitializeFeedFetchCopyInfo(session_state, *feeds_fetches_manager) == Status::OK());
}

}  // namespace training
}  // namespace onnxruntime
