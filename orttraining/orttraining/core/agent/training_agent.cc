// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/agent/training_agent.h"
#include "core/framework/utils.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/partial_graph_execution_state.h"
#include "core/framework/stream_execution_context.h"

namespace onnxruntime {
namespace training {

TrainingAgent::TrainingAgent(InferenceSession& session,
                             const std::vector<std::string>& fw_feed_names,
                             const std::vector<OrtDevice>& fw_outputs_device_info,
                             const std::vector<std::string>& bw_fetches_names,
                             const std::vector<OrtDevice>& bw_outputs_device_info,
                             int local_rank) : inference_session_(session) {
  ORT_UNUSED_PARAMETER(local_rank);
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  inference_session_.GetMemoryProfiler().GetMemoryInfo().SetLocalRank(local_rank);
#endif
  auto& session_state = session.GetSessionState();
  std::vector<std::string> fw_fetches_names;
  std::vector<std::string> bw_feed_names;

  size_t break_point = 0;
  auto* plan = session_state.GetExecutionPlan();
  auto& training_node_execution_order = plan->node_execution_order_in_training;
  for (auto node_index : training_node_execution_order) {
    if (session_state.GetKernel(node_index)->KernelDef().OpName() == "YieldOp") {
      auto& node = *(session_state.GetGraphViewer().GetGraph().GetNode(node_index));
      for (const auto& node_arg : node.InputDefs()) {
        fw_fetches_names.emplace_back(node_arg->Name());
      }

      for (const auto& node_arg : node.OutputDefs()) {
        bw_feed_names.emplace_back(node_arg->Name());
      }
      break;
    }
    break_point += 1;
  }

  fw_program_counter_end_ = break_point;
  bw_program_counter_end_ = training_node_execution_order.size();

  CreateAndInitializeFeedsFetchesManager(session_state, fw_feed_names, fw_fetches_names, fw_outputs_device_info,
                                         fw_feeds_fetches_manager_);

  CreateAndInitializeFeedsFetchesManager(session_state, bw_feed_names, bw_fetches_names, bw_outputs_device_info,
                                         bw_feeds_fetches_manager_);
}

TrainingAgent::~TrainingAgent() = default;

common::Status TrainingAgent::RunForward(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                         PartialGraphExecutionState& state, const OrtValueCachePtr& cache) {
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  inference_session_.GetMemoryProfiler().GetMemoryInfo().SetIteration(profile_step_);
  profile_step_ += 1;
#endif

  state.SetProgramCounterStart(0);
  state.SetProgramCounterEnd(fw_program_counter_end_);

  constexpr int32_t partial_graph_index = 0;
  return RunCore(feeds, fetches, state, *fw_feeds_fetches_manager_, cache, partial_graph_index);
}

common::Status TrainingAgent::RunBackward(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                          PartialGraphExecutionState& state) {
  state.SetProgramCounterStart(fw_program_counter_end_);
  state.SetProgramCounterEnd(bw_program_counter_end_);
  constexpr int32_t partial_graph_index = 1;
  return RunCore(feeds, fetches, state, *bw_feeds_fetches_manager_, nullptr, partial_graph_index);
}

common::Status TrainingAgent::RunCore(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                                      PartialGraphExecutionState& state, FeedsFetchesManager& feeds_fetches_manager,
                                      const OrtValueCachePtr& cache, int32_t partial_graph_index) {
  auto fetches_size = feeds_fetches_manager.GetFeedsFetchesInfo().output_names.size();
  fetches.resize(fetches_size, {});
  RunOptions run_options;
  return inference_session_.PartialRun(run_options, feeds, fetches, state, feeds_fetches_manager, cache,
                                       partial_graph_index);
}

void TrainingAgent::CreateAndInitializeFeedsFetchesManager(const SessionState& session_state,
                                                           const std::vector<std::string>& feed_names,
                                                           const std::vector<std::string>& fetches_names,
                                                           const std::vector<OrtDevice>& outputs_device_info,
                                                           std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager) {
  ORT_THROW_IF_ERROR(FeedsFetchesManager::Create(feed_names, fetches_names, session_state.GetOrtValueNameIdxMap(),
                                                 feeds_fetches_manager));
  auto& fetch_info = feeds_fetches_manager->GetMutableFetchesDeviceCopyInfo();
  for (size_t i = 0, end = fetches_names.size(); i < end; ++i) {
    fetch_info[i].target_device = outputs_device_info[i];
  }

  ORT_ENFORCE(utils::InitializeFeedFetchCopyInfo(session_state, *feeds_fetches_manager) == Status::OK());
}

}  // namespace training
}  // namespace onnxruntime
