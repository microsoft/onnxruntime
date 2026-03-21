// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>
#include <iostream>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace thread {

class QnnJobThreadPool {
 private:
  class QnnJobThread {
   public:
    QnnJobThread(uint8_t thread_num, QnnJobThreadPool* thread_pool_ptr)
      : thread_num_(thread_num), tp_(thread_pool_ptr){
      LOGS_DEFAULT(VERBOSE) << "Thread " << std::to_string(thread_num_) << " created";
    }

    ~QnnJobThread() {
      Stop();
    }

    void Start() {
      if (!thread_ && IsStopped()) {
        {
          std::unique_lock<std::mutex> lock(thread_stopped_mutex_);
          thread_stopped_ = false;
        }

        thread_ = std::make_shared<std::thread>([this]() {
          do {
            auto job = tp_->GetJobFromQueueIfExists(thread_num_);
            if (job) {
              {
                std::unique_lock<std::mutex> lock(thread_active_mutex_);
                thread_active_ = true;
              }

              thread_activity_change_cv_.notify_all();
              tp_->NotifyJobStarted();

              job();

              {
                std::unique_lock<std::mutex> lock(thread_active_mutex_);
                thread_active_ = false;
              }

              thread_activity_change_cv_.notify_all();
            } else {
              tp_->WaitForJobQueueUpdate(thread_num_);
            }
          } while (!IsStopped());
        });

        LOGS_DEFAULT(VERBOSE) << "Thread " << std::to_string(thread_num_) << " started";
      }
    }

    void Stop() {
      if (thread_ && !IsStopped()) {
        {
          std::unique_lock<std::mutex> lock(thread_stopped_mutex_);
          thread_stopped_ = true;
        }
        thread_->join();
        thread_.reset();

        LOGS_DEFAULT(VERBOSE) << "Thread " << std::to_string(thread_num_) << " stopped";
      }
    }

    bool IsActive() {
      std::unique_lock<std::mutex> lock(thread_active_mutex_);
      return thread_active_;
    }

    void WaitUntilInactive() {
      if (IsActive()) {
        std::unique_lock<std::mutex> lock(thread_active_mutex_);
        thread_activity_change_cv_.wait(lock, [this]() {
          return !thread_active_;
        });
      }
    }

   private:

    bool IsStopped() {
      std::unique_lock<std::mutex> lock(thread_stopped_mutex_);
      return thread_stopped_;
    }

    const uint8_t thread_num_;
    QnnJobThreadPool* const tp_;

    std::condition_variable thread_activity_change_cv_;

    std::mutex thread_active_mutex_;
    std::mutex thread_stopped_mutex_;
    bool thread_active_ = false;
    bool thread_stopped_ = true;

    std::shared_ptr<std::thread> thread_;
  };

 public:
  QnnJobThreadPool(uint8_t max_num_threads): max_num_threads_(max_num_threads), running_(false) {
    thread_pool_.reserve(max_num_threads);
  }

  ~QnnJobThreadPool() {
    Stop();
  }

  void Start() {
    if (IsRunning()) {
      return;
    }

    LOGS_DEFAULT(VERBOSE) << "Start";
    std::unique_lock<std::mutex> s_lock(state_mutex_);
    std::unique_lock<std::mutex> tp_lock(thread_pool_mutex_);
    running_ = true;

    while (thread_pool_.size() < max_num_threads_ && !job_queue_.empty()) {
      StartJobThread();
    }
  }

  void Stop() {
    if (!IsRunning()) {
      return;
    }
    {
      LOGS_DEFAULT(VERBOSE) << "Stop";
      std::unique_lock<std::mutex> s_lock(state_mutex_);
      running_ = false;
    }

    // Wake up all waiting threads
    job_submitted_cv_.notify_all();
  }

  void WaitForAllJobsToFinish() {
    if (!IsRunning()) {
      return;
    }
    
    LOGS_DEFAULT(VERBOSE) << "Waiting for all jobs to finish before stopping";

    if (!job_queue_.empty()) {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      job_started_cv_.wait(lock, [this]() {
        return job_queue_.empty();
      });
    }

    for (auto& t : thread_pool_) {
      t->WaitUntilInactive();
    }

    LOGS_DEFAULT(VERBOSE) << "Done waiting on all jobs" << std::endl;
  }

  void Submit(std::function<void()> job) {
    LOGS_DEFAULT(VERBOSE) << "Job submitted";
    {
      std::unique_lock<std::mutex> lock(thread_pool_mutex_);
      LOGS_DEFAULT(VERBOSE) << "Number of active threads: " << std::to_string(thread_pool_.size());
      if (thread_pool_.size() < max_num_threads_ && IsRunning()) {
        StartJobThread();
      }
    }

    std::unique_lock<std::mutex> lock(queue_mutex_);
    job_queue_.push(std::move(job));
    LOGS_DEFAULT(VERBOSE) << "Job pushed to queue, current size: " << std::to_string(job_queue_.size());
    job_submitted_cv_.notify_one();
  }

 private:

   void StartJobThread() {
     uint8_t thread_num = static_cast<uint8_t>(thread_pool_.size());
     auto job_thread = std::make_unique<QnnJobThread>(thread_num, this);
     job_thread->Start();
     thread_pool_.push_back(std::move(job_thread));
   }

   void WaitForJobQueueUpdate(const uint8_t thread_num) {
     std::unique_lock<std::mutex> lock(queue_mutex_);
     LOGS_DEFAULT(VERBOSE) << "Thread " << std::to_string(thread_num) << " waiting for a job";
     job_submitted_cv_.wait(lock, [this] {
       return !job_queue_.empty() || !IsRunning();
     });
   }

   std::function<void()> GetJobFromQueueIfExists(const uint8_t thread_num) {
     std::unique_lock<std::mutex> lock(queue_mutex_);
     LOGS_DEFAULT(VERBOSE) << "Thread " << std::to_string(thread_num) << " checking for job, queue size: " << std::to_string(job_queue_.size());
     if (!job_queue_.empty()) {
       auto job = job_queue_.front();
       job_queue_.pop();
       return job;
     }

     return nullptr;
   }

   void NotifyJobStarted() {
     job_started_cv_.notify_all();
   }

   bool IsRunning() {
     std::unique_lock<std::mutex> lock(state_mutex_);
     return running_;
   }

 std::condition_variable job_submitted_cv_;
 std::condition_variable job_started_cv_;

 std::mutex queue_mutex_;
 std::queue<std::function<void()>> job_queue_;

 std::mutex thread_pool_mutex_;
 std::vector<std::unique_ptr<QnnJobThread>> thread_pool_;

 const uint8_t max_num_threads_;

 std::mutex state_mutex_;
 bool running_;
};

}
}
}