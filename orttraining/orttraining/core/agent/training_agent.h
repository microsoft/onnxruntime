// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <thread>
#include <future>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/framework_common.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "orttraining/training_ops/cpu/controlflow/ort_tasks.h"


namespace onnxruntime {
namespace training {
class IOBinding;

// TODO: Define how minimal builds will play out here!
class TrainingAgent {
 public:

  explicit TrainingAgent(InferenceSession* session);

  virtual ~TrainingAgent();


  // For ORTModule.forward()
  virtual common::Status RunInBackgroundAndWaitForYield(const RunOptions& run_options, onnxruntime::IOBinding& io_binding,
                                                        std::vector<OrtValue>& user_outputs,
                                                        int64_t& run_id) ORT_MUST_USE_RESULT;

  // For ORTModule.backward()
  common::Status ContinueRunInBackground(int64_t run_id, const std::vector<OrtValue>& backward_output_grads) ORT_MUST_USE_RESULT;

  void CancelBackgroundTask(int64_t run_id);

  InferenceSession* GetSessionHandle();

  protected:
	// mutex for accessing bg_threads_
  std::mutex bg_threads_mutex_;

  // background threads for RunInBackgroundAndWaitForYield and ContinueRunInBackground
  std::unordered_map<int64_t, std::thread> bg_threads_;

  // TrainingAgent runs on a InferenceSession under the hood
  std::unique_ptr<InferenceSession> inference_session_;

};


}  // namespace training
}  // namespace onnxruntime
