// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/agent/training_agent.h"
#include "core/session/IOBinding.h"
#include "orttraining/training_ops/cpu/controlflow/ort_tasks.h"

namespace onnxruntime {
namespace training {

TrainingAgent::TrainingAgent(InferenceSession* session) : inference_session_(session) {}

TrainingAgent::~TrainingAgent() {
  // TODO: Properly cancel outstanding background tasks
  // Following implementation only handle the case where bg_thread is waiting for backward inputs
  // Background thread can also be in other states, such as running Forward() or running Backward()
  std::vector<int64_t> run_ids;
  {
    std::lock_guard<std::mutex> lock(bg_threads_mutex_);
    for (auto it = bg_threads_.begin(); it != bg_threads_.end(); ++it) {
      run_ids.push_back(it->first);
    }
  }
  for (int64_t run_id : run_ids) {
    if (!onnxruntime::contrib::OrtTasks::GetInstance().TaskIsCompleted(run_id)) {
      CancelPendingBackwardRun(run_id);
    }
  }
};

common::Status TrainingAgent::RunForward(const RunOptions& run_options, onnxruntime::IOBinding& io_binding,
                                         std::vector<OrtValue>& user_outputs, int64_t& run_id,
                                         int32_t& token_out) {
  std::promise<void> setup_promise;
  std::future<void> setup_future = setup_promise.get_future();

  // Passing run_options and io_binding by reference to the bg_thread,
  // this is ok because they are ORTModule's member, and they are presistent through forward and backward calls
  auto bg_thread = std::thread([this](std::future<void> setup_future, const RunOptions& run_options, onnxruntime::IOBinding& io_binding) {
    // wait until task is properly setup
    setup_future.get();

    common::Status status = inference_session_->Run(run_options, io_binding);

    onnxruntime::contrib::OrtTasks::GetInstance().CompleteTask(status, onnxruntime::contrib::TOKEN_END_BACKWARD);

    // // If training outputs still hasn't been consumed at this point, i.e. forward + backward hasn't complete itself
    // // this indicates that Run() call returned before hitting YieldOp, due to hitting some exception during the execution
    // // In this case, we need to wake up the foreground thread and pass along the failed status.
    // // Otherwise, foreground thread will be stuck waiting for forward_outputs.
    // if (onnxruntime::contrib::OrtTasks::GetInstance().KernelOutputsIsValid()) {
    //   ORT_ENFORCE(!status.IsOK());
    //   // signal main thread for background thread completion
    //   onnxruntime::contrib::OrtTasks::GetInstance().SetKernelOutputs(status, {});
    // }
  },
                               std::move(setup_future), std::cref(run_options), std::ref(io_binding));

  run_id = std::hash<std::thread::id>()(bg_thread.get_id());
  {
    std::lock_guard<std::mutex> lock(bg_threads_mutex_);
    bg_threads_[run_id] = std::move(bg_thread);
  }

  onnxruntime::contrib::OrtTasks::GetInstance().CreateBackgroundTask(run_id);

  LOGS(*inference_session_->GetLogger(), VERBOSE) << "InferenceSession::Forward() call created a task with run_id " << run_id;

  // background task is setup, unblock background thread to continue
  setup_promise.set_value();

  // Wait for data/signal from
  // 1. Yield op, if the bg thread sucessfully reached Yield's signal point
  // 2. The end of bg thread, if it hit execptions and returned earlier
  auto forward_outputs = onnxruntime::contrib::OrtTasks::GetInstance().WaitForKernelOutputs(run_id);
  const Status& forward_status = forward_outputs.status;
  user_outputs = std::move(forward_outputs.values);
  token_out = forward_outputs.token_id;

  // background thread has completed incorrectly.
  if (!forward_status.IsOK()) {
    std::thread thread;
    {
      std::lock_guard<std::mutex> lock(bg_threads_mutex_);
      std::swap(thread, bg_threads_[run_id]);
      bg_threads_.erase(run_id);
    }
    ORT_ENFORCE(thread.joinable());
    thread.join();
    onnxruntime::contrib::OrtTasks::GetInstance().RemoveTask(run_id);
    return forward_status;
  }

  return Status::OK();
}

common::Status TrainingAgent::ResumeForward(int64_t run_id, const std::vector<OrtValue>& resumed_inputs,
                                            std::vector<OrtValue>& user_outputs, int32_t& token_out) {
  LOGS(*inference_session_->GetLogger(), VERBOSE) << "Running TrainingAgent::ResumeRunForward() with run_id " << run_id;

  // resume background thread
  onnxruntime::contrib::OrtTasks::GetInstance().SetExternalKernelOutputs(run_id, resumed_inputs);

  // Wait for data/signal from
  // 1. Yield op, if the bg thread sucessfully reached Yield's signal point
  // 2. The end of bg thread, if it hit execptions and returned earlier
  auto forward_outputs = onnxruntime::contrib::OrtTasks::GetInstance().WaitForKernelOutputs(run_id);
  const Status& forward_status = forward_outputs.status;
  user_outputs = std::move(forward_outputs.values);
  token_out = forward_outputs.token_id;
  // background thread has completed incorrectly.
  if (!forward_status.IsOK()) {
    std::thread thread;
    {
      std::lock_guard<std::mutex> lock(bg_threads_mutex_);
      std::swap(thread, bg_threads_[run_id]);
      bg_threads_.erase(run_id);
    }
    ORT_ENFORCE(thread.joinable());
    thread.join();
    onnxruntime::contrib::OrtTasks::GetInstance().RemoveTask(run_id);
    return forward_status;
  }

  return Status::OK();
}

common::Status TrainingAgent::RunBackward(int64_t run_id, const std::vector<OrtValue>& backward_output_grads,
                                          std::vector<OrtValue>& user_outputs, int32_t& token_out) {
  LOGS(*inference_session_->GetLogger(), VERBOSE) << "Running TrainingAgent::Backward() with run_id " << run_id;

  // resume background thread
  onnxruntime::contrib::OrtTasks::GetInstance().SetExternalKernelOutputs(run_id, backward_output_grads, false);

  auto outputs = onnxruntime::contrib::OrtTasks::GetInstance().WaitForKernelOutputs(run_id, token_out);
  token_out = outputs.token_id;
  user_outputs = std::move(outputs.values);
  // Status bg_thread_status = onnxruntime::contrib::OrtTasks::GetInstance().WaitForStatus(run_id);
  if (token_out == TOKEN_END_BACKWARD) {  // all training compute completed.
    std::thread bg_thread;
    {
      std::lock_guard<std::mutex> lock(bg_threads_mutex_);
      std::swap(bg_thread, bg_threads_[run_id]);
      bg_threads_.erase(run_id);
    }

    // wait for bg_thread to complete
    ORT_ENFORCE(bg_thread.joinable());
    bg_thread.join();
    onnxruntime::contrib::OrtTasks::GetInstance().RemoveTask(run_id);
  }

  return bg_thread_status;
}

void TrainingAgent::CancelPendingBackwardRun(int64_t run_id) {
  LOGS(*inference_session_->GetLogger(), INFO) << "Canceling background task with run_id " << run_id;

  // resume background thread with terminate = true
  onnxruntime::contrib::OrtTasks::GetInstance().SetExternalKernelOutputs(run_id, {}, true);

  // wait for bg_thread to complete
  std::thread bg_thread;
  {
    std::lock_guard<std::mutex> lock(bg_threads_mutex_);
    std::swap(bg_thread, bg_threads_[run_id]);
    bg_threads_.erase(run_id);
  }
  ORT_ENFORCE(bg_thread.joinable());
  bg_thread.join();
  onnxruntime::contrib::OrtTasks::GetInstance().RemoveTask(run_id);
}

}  // namespace training
}  // namespace onnxruntime
