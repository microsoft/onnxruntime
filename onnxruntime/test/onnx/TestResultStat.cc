// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdio>
#include <sstream>
#include <algorithm>
#include <vector>
#include "TestResultStat.h"

namespace {

void operator<<(std::ostringstream& oss, const std::pair<std::string, std::string>& p) {
  oss << p.first << " of " << p.second;
}

template <typename T1>
std::string containerToStr(const T1& input) {
  std::ostringstream oss;
  bool is_first = true;
  std::vector<typename T1::value_type> vec(input.begin(), input.end());
  std::sort(vec.begin(), vec.end());
  for (const auto& s : vec) {
    if (!is_first) oss << ", ";
    oss << s;
    is_first = false;
  }
  return oss.str();
}
}  // namespace

std::string TestResultStat::ToString() {
  std::string not_implemented_kernels_str = containerToStr(this->not_implemented_kernels);
  std::string failed_kernels_str = containerToStr(this->failed_kernels);
  int failed = static_cast<int>(this->total_test_case_count) - this->succeeded - this->skipped - this->not_implemented;
  int other_reason_failed = failed - this->load_model_failed - this->result_differs - this->throwed_exception - this->invalid_graph;
  std::ostringstream oss;
  oss << "result: "
         "\n\tModels: "
      << this->total_model_count
      << "\n\tTotal test cases: "
      << this->total_test_case_count
      << "\n\t\tSucceeded: " << this->succeeded
      << "\n\t\tNot implemented: " << this->not_implemented
      << "\n\t\tFailed: " << failed << "\n";
  if (this->invalid_graph)
    oss << "\t\t\tGraph is invalid:" << this->invalid_graph << "\n";
  if (this->load_model_failed)
    oss << "\t\t\tGot exception while loading model: " << this->load_model_failed << "\n";
  if (this->throwed_exception)
    oss << "\t\t\tGot exception while running: " << this->throwed_exception << "\n";
  if (this->result_differs)
    oss << "\t\t\tResult differs: " << this->result_differs << "\n";
  if (other_reason_failed != 0) oss << "\t\t\tOther reason:" << other_reason_failed << "\n";
  oss << "\tStats by Operator type:\n";
  oss << "\t\tNot implemented(" << this->not_implemented_kernels.size() << "): " << not_implemented_kernels_str << "\n\t\tFailed:"
      << failed_kernels_str << "\n";
  oss << "Failed Test Cases:" << containerToStr(failed_test_cases) << "\n";
  return oss.str();
}
