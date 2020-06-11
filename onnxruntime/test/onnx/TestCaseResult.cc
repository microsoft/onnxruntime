// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TestCaseResult.h"

void TestCaseResult::SetResult(size_t task_id, EXECUTE_RESULT r) {
  std::lock_guard<std::mutex> guard(result_mutex_);
  if (execution_result_[task_id] == EXECUTE_RESULT::NOT_SET) {
    execution_result_[task_id] = r;
  } else if (r != EXECUTE_RESULT::SUCCESS && execution_result_[task_id] == EXECUTE_RESULT::SUCCESS) {
    // store first failure
    execution_result_[task_id] = r;
  }
}
