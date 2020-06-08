// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <vector>
#include <core/platform/env_time.h>
#include <cstring>
#include <mutex>

//result of a single test run: 1 model with 1 test dataset
enum class EXECUTE_RESULT {
  NOT_SET = 0,
  SUCCESS = 1,
  UNKNOWN_ERROR = -1,
  WITH_EXCEPTION = -2,
  RESULT_DIFFERS = -3,
  SHAPE_MISMATCH = -4,
  TYPE_MISMATCH = -5,
  NOT_SUPPORT = -6,
  LOAD_MODEL_FAILED = -7,
  INVALID_GRAPH = -8,
  INVALID_ARGUMENT = -9,
  MODEL_SHAPE_MISMATCH = -10,
  MODEL_TYPE_MISMATCH = -11,
};

class TestCaseResult {
 public:
  TestCaseResult(size_t count, EXECUTE_RESULT result, const std::string& node_name1)
      : node_name(node_name1), execution_result_(count, result) {
    onnxruntime::SetTimeSpecToZero(&spent_time_);
  }

  void SetResult(size_t task_id, EXECUTE_RESULT r);

  const std::vector<EXECUTE_RESULT>& GetExcutionResult() const {
    return execution_result_;
  }

  //Time spent in Session::Run. It only make sense when SeqTestRunner was used
  onnxruntime::TIME_SPEC GetSpentTime() const {
    return spent_time_;
  }

  //Time spent in Session::Run. It only make sense when SeqTestRunner was used
  void SetSpentTime(const onnxruntime::TIME_SPEC& input) const {
    memcpy((void*)&spent_time_, &input, sizeof(input));
  }

  //only valid for single node tests;
  const std::string node_name;

 private:
  onnxruntime::TIME_SPEC spent_time_;
  std::vector<EXECUTE_RESULT> execution_result_;
  std::mutex result_mutex_;
};
