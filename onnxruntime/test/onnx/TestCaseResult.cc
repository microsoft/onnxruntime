// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TestCaseResult.h"

void TestCaseResult::SetResult(size_t task_id, EXECUTE_RESULT r) {
  excution_result_[task_id] = r;
}
