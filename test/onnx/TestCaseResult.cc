// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TestCaseResult.h"
#include "core/common/common.h"
#include <iostream>

void TestCaseResult::SetResult(size_t task_id, EXECUTE_RESULT r) {
  ORT_ENFORCE(task_id < execution_result_.size(), "Index out of bounds");
  if (execution_result_[task_id] == EXECUTE_RESULT::NOT_SET) {
    execution_result_[task_id] = r;
  } else if (r != EXECUTE_RESULT::SUCCESS && execution_result_[task_id] == EXECUTE_RESULT::SUCCESS) {
    // store first failure
    execution_result_[task_id] = r;
  }
}

std::ostream& operator<<(std::ostream& os, EXECUTE_RESULT result) {
  return os << "EXECUTION_RESULT: " << static_cast<int>(result);
}
