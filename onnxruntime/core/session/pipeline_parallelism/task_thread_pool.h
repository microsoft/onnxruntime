/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIOONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Modifications Copyright (c) Microsoft.

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <iostream>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
class TaskThreadPool {
 private:
  struct task_element_t {
    bool run_with_id;
    std::function<void()> no_id;
    std::function<void(std::size_t)> with_id;

    task_element_t(task_element_t&& other) {
      run_with_id = other.run_with_id;
      no_id = std::move(other.no_id);
      with_id = std::move(other.with_id);
    }

    explicit task_element_t(std::function<void()>&& f)
        : run_with_id(false), no_id(std::move(f)) {}

    explicit task_element_t(std::function<void(std::size_t)>&& f)
        : run_with_id(true), with_id(std::move(f)) {}
  };

  std::queue<task_element_t> tasks_;
  std::vector<std::thread> threads_;
  OrtMutex mutex_;
  OrtCondVar condition_;
  OrtCondVar completed_;
  bool running_;
  bool complete_;
  std::size_t available_;
  std::size_t total_;

 public:
  /// @brief Constructor.
  explicit TaskThreadPool(std::size_t pool_size)
      : threads_(pool_size), running_(true), complete_(true), available_(pool_size), total_(pool_size) {
    for (std::size_t i = 0; i < pool_size; ++i) {
      threads_[i] = std::thread(std::bind(&TaskThreadPool::MainLoop, this, i));
    }
  }

  /// @brief Destructor.
  ~TaskThreadPool() {
    // Set running flag to false then notify all threads.
    {
      std::unique_lock<OrtMutex> lock(mutex_);
      running_ = false;
      condition_.notify_all();
    }

    try {
      for (auto& t : threads_) {
        t.join();
      }
    }
    // Suppress all exceptions.
    catch (const std::exception& ex) {
      std::cerr << "Exception joining threads in TaskThreadPool: " << ex.what();
    }
  }

  void RunTask(std::function<void()>&& task) {
    std::unique_lock<OrtMutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.push(task_element_t(std::move(task)));
    complete_ = false;
    condition_.notify_one();
  }

  void RunTaskWithID(std::function<void(std::size_t)>&& task) {
    std::unique_lock<OrtMutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.push(task_element_t(std::move(task)));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief Wait for queue to be empty
  void WaitWorkComplete() {
    std::unique_lock<OrtMutex> lock(mutex_);
    while (!complete_)
      completed_.wait(lock);
  }

 private:
  // ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TaskThreadPool);

  /// @brief Entry point for pool threads.
  void MainLoop(std::size_t index) {
    while (running_) {
      // Wait on condition variable while the task is empty and
      // the pool is still running.
      std::unique_lock<OrtMutex> lock(mutex_);
      while (tasks_.empty() && running_) {
        condition_.wait(lock);
      }

      // If pool is no longer running, break out of loop.
      if (!running_) break;

      // Copy task locally and remove from the queue.  This is
      // done within its own scope so that the task object is
      // destructed immediately after running the task.  This is
      // useful in the event that the function contains
      // shared_ptr arguments bound via bind.
      {
        auto task = std::move(tasks_.front());
        tasks_.pop();
        // Decrement count, indicating thread is no longer available.
        --available_;

        lock.unlock();

        // Run the task.
        try {
          if (task.run_with_id) {
            task.with_id(index);
          } else {
            task.no_id();
          }
        } catch (const std::exception& ex) {
          LOGS_DEFAULT(ERROR) << "Exception running TaskThreadPool task: " << ex.what();
        } catch (...) {
          LOGS_DEFAULT(ERROR) << "Unknown exception running TaskThreadPool task";
        }

        // Update status of empty, maybe
        // Need to recover the lock first
        lock.lock();

        // Increment count, indicating thread is available.
        ++available_;
        if (tasks_.empty() && available_ == total_) {
          complete_ = true;
          completed_.notify_one();
        }
      }
    }  // while running_
  }
};
}  // namespace onnxruntime
