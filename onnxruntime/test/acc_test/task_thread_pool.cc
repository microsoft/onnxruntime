// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "task_thread_pool.h"
#include <mutex>
#include "acc_task.h"

TaskThreadPool::TaskThreadPool(size_t num_threads) {
  threads_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; i++) {
    threads_.emplace_back(&TaskThreadPool::ThreadEntry, this);
  }
}

TaskThreadPool::~TaskThreadPool() {
  {
    // Acquire lock, set shutdown to true, and wake up all threads.
    std::unique_lock<std::mutex> lock(lock_);
    shutdown_ = true;
    signal_.notify_all();
  }

  // Wait for all threads to exit.
  for (auto& thread : threads_) {
    thread.join();
  }
}

void TaskThreadPool::CompleteTasks(Span<Task> tasks) {
  // Assert that it is only possible to call CompleteTasks() when either
  // this is the first set of tasks or we've completely processed the previous tasks.
  assert(tasks_completed_ == tasks_.size());

  {
    // Acquire lock, set new tasks, and wake up all threads.
    std::unique_lock<std::mutex> lock(lock_);
    tasks_ = tasks;
    tasks_completed_ = 0;
    next_task_index_ = 0;
    signal_.notify_all();
  }

  // The main thread (calling thread) can also help out until all tasks have been completed.
  while (tasks_completed_ < tasks_.size()) {
    while (RunNextTask()) {
      // Keep helping out the pool threads.
    }
  }
}

void TaskThreadPool::ThreadEntry() {
  while (true) {
    // Keep running tasks until they have *all* been claimed by some thread.
    while (RunNextTask()) {
    }

    {
      // Get lock and sleep if all tasks have been taken by some thread. If shutdown_ flag is set, exit.
      std::unique_lock<std::mutex> lock(lock_);
      while (!shutdown_ && (next_task_index_ >= tasks_.size())) {
        signal_.wait(lock);  // wait() may be unblocked spuriously (according to docs), so need to call it in a loop.
      }

      if (shutdown_) {
        return;
      }
    }
  }
}

bool TaskThreadPool::RunNextTask() {
  if (tasks_.empty()) {
    return false;
  }

  const size_t task_index = std::atomic_fetch_add(&next_task_index_, 1);
  if (task_index >= tasks_.size()) {
    return false;
  }

  tasks_[task_index].Run();

  std::atomic_fetch_add(&tasks_completed_, 1);
  return true;
}
