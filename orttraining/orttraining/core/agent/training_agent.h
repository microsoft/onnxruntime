// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <thread>
#include <future>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/framework_common.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
struct PartialGraphExecutionState;
typedef InlinedHashMap<std::string, OrtValue> OrtValueCache;
typedef std::shared_ptr<OrtValueCache> OrtValueCachePtr;

namespace training {

class TrainingAgent {
 public:
  explicit TrainingAgent(InferenceSession& session,
                         const std::vector<std::string>& fw_feed_names,
                         const std::vector<OrtDevice>& fw_outputs_device_info,
                         const std::vector<std::string>& bw_fetches_names,
                         const std::vector<OrtDevice>& bw_outputs_device_info,
                         int local_rank = 0);
  ~TrainingAgent();
  // For ORTModule.forward()
  [[nodiscard]] common::Status RunForward(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                            PartialGraphExecutionState& state, const OrtValueCachePtr& cache);

  // For ORTModule.backward()
  [[nodiscard]] common::Status RunBackward(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                             PartialGraphExecutionState& state);

  [[nodiscard]] common::Status RunCore(const std::vector<OrtValue>& feeds, std::vector<OrtValue>& fetches,
                         PartialGraphExecutionState& state, FeedsFetchesManager& feeds_fetches_manager,
                         const OrtValueCachePtr& cache, int32_t partial_graph_index);

  void CreateAndInitializeFeedsFetchesManager(const SessionState& session_state,
                                              const std::vector<std::string>& feed_names,
                                              const std::vector<std::string>& fetches_names,
                                              const std::vector<OrtDevice>& outputs_device_info,
                                              std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager);

 private:
  // TrainingAgent runs on a InferenceSession under the hood
  InferenceSession& inference_session_;
  std::unique_ptr<FeedsFetchesManager> fw_feeds_fetches_manager_;
  std::unique_ptr<FeedsFetchesManager> bw_feeds_fetches_manager_;
  size_t fw_program_counter_end_;
  size_t bw_program_counter_end_;

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  size_t profile_step_{0};
#endif
};

}  // namespace training
}  // namespace onnxruntime
