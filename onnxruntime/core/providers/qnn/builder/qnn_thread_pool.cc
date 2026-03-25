// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_thread_pool.h"

namespace onnxruntime {
namespace qnn {
namespace thread {

QnnJobThreadPool::QnnJobThread::QnnJobThread(uint8_t thread_num, QnnJobThreadPool* thread_pool_ptr)
    : thread_num_(thread_num), tp_(thread_pool_ptr) {
  LOGS_DEFAULT(VERBOSE) << "QnnJobThread: Thread " << std::to_string(thread_num_) << " created";

  // Used to exit out of QnnJobThreadPool::WaitForJobQueueUpdate() regardless of job queue status
  exit_predicate_ = [this]() {
    return IsStopped();
  };
}

QnnJobThreadPool::QnnJobThread::~QnnJobThread() {
  Stop();
}

void QnnJobThreadPool::QnnJobThread::Start() {
  // Only created thread if no thread exists and the current state is stopped
  if (!thread_ && IsStopped()) {
    std::unique_lock<std::mutex> lock(thread_state_mutex_);
    thread_stopped_ = false;
  } else {
    return;
  }

  thread_ = std::make_unique<std::thread>([this]() {
    do {
      auto job = tp_->GetJobFromQueueIfExists(thread_num_);
      if (job) {
        SetActive();
        tp_->NotifyJobStarted();

        job();

        SetInactive();
      } else {
        tp_->WaitForJobQueueUpdate(thread_num_, exit_predicate_);
      }
    } while (!IsStopped());
  });

  LOGS_DEFAULT(VERBOSE) << "QnnJobThread: Thread " << std::to_string(thread_num_) << " started";
}

void QnnJobThreadPool::QnnJobThread::Stop() {
  // Only stop if the thread exists and current state is running
  if (thread_ && !IsStopped()) {
    LOGS_DEFAULT(VERBOSE) << "QnnJobThread: Thread " << std::to_string(thread_num_) << " stopping";
    std::unique_lock<std::mutex> lock(thread_state_mutex_);
    thread_stopped_ = true;
  } else {
    return;
  }

  thread_->join();
  thread_.reset();

  LOGS_DEFAULT(VERBOSE) << "QnnJobThread: Thread " << std::to_string(thread_num_) << " stopped";
}

void QnnJobThreadPool::QnnJobThread::WaitUntilInactive() {
  std::unique_lock<std::mutex> lock(thread_activity_mutex_);
  if (thread_active_) {
    thread_activity_change_cv_.wait(lock, [this]() {
      return !thread_active_;
    });
  }
}

QnnJobThreadPool::QnnJobThreadPool(uint8_t max_num_threads)
  : max_num_threads_(max_num_threads), running_(false) {
  thread_pool_.reserve(max_num_threads);
}

QnnJobThreadPool::~QnnJobThreadPool() {
  Stop();
  WaitForAllJobsToFinish();
}

void QnnJobThreadPool::Start() {
  if (IsRunning()) {
    return;
  }

  LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Start";
  std::unique_lock<std::mutex> s_lock(state_mutex_);
  std::unique_lock<std::mutex> tp_lock(thread_pool_mutex_);
  running_ = true;

  while (thread_pool_.size() < max_num_threads_) {
    StartJobThread();
  }
}


void QnnJobThreadPool::Stop() {
  if (!IsRunning()) {
    return;
  }

  LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Stop";
  std::unique_lock<std::mutex> s_lock(state_mutex_);
  running_ = false;

  for (auto& thread : thread_pool_) {
    thread->Stop();
  }
}

void QnnJobThreadPool::WaitForAllJobsToFinish() {
  LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Waiting for all jobs to finish";

  // Block all newly submitted jobs from entering the queue
  std::unique_lock<std::mutex> lock(queue_mutex_);
  // Only wait until queue is empty if thread pool has not been stopped
  if (IsRunning() && !job_queue_.empty()) {
    job_started_cv_.wait(lock, [this]() {
      return job_queue_.empty();
    });
  }

  for (auto& t : thread_pool_) {
    t->WaitUntilInactive();
  }

  LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Done waiting on all jobs" << std::endl;
}

void QnnJobThreadPool::SubmitJob(std::function<void()> job) {
  LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Job submitted";

  std::unique_lock<std::mutex> lock(queue_mutex_);
  job_queue_.push(std::move(job));
  LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Job pushed to queue, current size: " << std::to_string(job_queue_.size());
  job_submitted_cv_.notify_one();
}

void QnnJobThreadPool::StartJobThread() {
  uint8_t thread_num = static_cast<uint8_t>(thread_pool_.size());
  auto job_thread = std::make_unique<QnnJobThread>(thread_num, this);
  job_thread->Start();
  thread_pool_.push_back(std::move(job_thread));
}

void QnnJobThreadPool::WaitForJobQueueUpdate(const uint8_t thread_num, std::function<bool()>& exit_predicate) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Thread " << std::to_string(thread_num) << " waiting for a job";
  job_submitted_cv_.wait_for(lock, std::chrono::milliseconds(200), [this, &exit_predicate] {
    return !job_queue_.empty() || exit_predicate();
  });
}

std::function<void()> QnnJobThreadPool::GetJobFromQueueIfExists(const uint8_t thread_num) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Thread " << std::to_string(thread_num) << " checking for job, queue size: " << std::to_string(job_queue_.size());
  if (!job_queue_.empty()) {
    LOGS_DEFAULT(VERBOSE) << "QnnJobThreadPool: Thread " << std::to_string(thread_num) << " received a job";
    auto job = job_queue_.front();
    job_queue_.pop();
    return job;
  }

  return nullptr;
}

}  // namespace thread
}  // namespace qnn
}  // namespace onnxruntime