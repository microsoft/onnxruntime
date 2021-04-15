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

class TrainingAgent {

  public:
    explicit TrainingAgent(InferenceSession& session);
    ~TrainingAgent();
    // For ORTModule.forward()
    common::Status RunForward(const RunOptions& run_options, onnxruntime::IOBinding& io_binding,
                                               std::vector<OrtValue>& user_outputs,
                                               int64_t& run_id) ORT_MUST_USE_RESULT;
    // For ORTModule.backward()
    common::Status RunBackward(int64_t run_id, const std::vector<OrtValue>& backward_output_grads) ORT_MUST_USE_RESULT;
    void CancelPendingBackwardRun(int64_t run_id);

  private:
    // mutex for accessing bg_threads_
    std::mutex bg_threads_mutex_;
    // background threads for RunForward and RunBackward
    std::unordered_map<int64_t, std::thread> bg_threads_;
    // TrainingAgent runs on a InferenceSession under the hood
    InferenceSession& inference_session_;
};

}  // namespace training
}  // namespace onnxruntime
