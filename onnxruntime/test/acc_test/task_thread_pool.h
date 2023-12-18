// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <condition_variable>
#include <mutex>
#include <vector>

#include "basic_utils.h"
#include "acc_task.h"

class TaskThreadPool {
 public:
  TaskThreadPool(int num_threads);
  ~TaskThreadPool();

  void CompleteTasks(Span<Task> tasks);

 private:
  void ThreadEntry();
  bool RunNextTask();

  std::mutex lock_;
  std::condition_variable signal_;
  bool shutdown_ = false;
  Span<Task> tasks_;
  std::atomic<size_t> next_task_index_ = 0;
  std::atomic<size_t> tasks_completed_ = 0;
  std::vector<std::thread> threads_;
};
