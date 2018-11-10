// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include "core/common/logging/isink.h"
#include "TestCaseResult.h"
#include <CppUnitTest.h>

class VsTestSink : public onnxruntime::logging::ISink {
 public:
  void SendImpl(const onnxruntime::logging::Timestamp& timestamp, const std::string& logger_id_, const onnxruntime::logging::Capture& message) override;
};

namespace Microsoft {
namespace VisualStudio {
namespace CppUnitTestFramework {
template <>
std::wstring ToString<>(const EXECUTE_RESULT& q) {
  switch (q) {
    case EXECUTE_RESULT::SUCCESS:
      return L"SUCCESS";
    case EXECUTE_RESULT::UNKNOWN_ERROR:
      return L"UNKNOWN_ERROR";
    case EXECUTE_RESULT::WITH_EXCEPTION:
      return L"WITH_EXCEPTION";
    case EXECUTE_RESULT::RESULT_DIFFERS:
      return L"RESULT_DIFFERS";
    case EXECUTE_RESULT::SHAPE_MISMATCH:
      return L"SHAPE_MISMATCH";
    case EXECUTE_RESULT::TYPE_MISMATCH:
      return L"TYPE_MISMATCH";
    case EXECUTE_RESULT::NOT_SUPPORT:
      return L"NOT_SUPPORT";
    case EXECUTE_RESULT::LOAD_MODEL_FAILED:
      return L"LOAD_MODEL_FAILED";
    case EXECUTE_RESULT::INVALID_GRAPH:
      return L"INVALID_GRAPH";
    case EXECUTE_RESULT::INVALID_ARGUMENT:
      return L"INVALID_ARGUMENT";
    case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
      return L"MODEL_SHAPE_MISMATCH";
    case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
      return L"MODEL_TYPE_MISMATCH";
  }
  return L"UNKNOWN_RETURN_CODE";
}
}  // namespace CppUnitTestFramework
}  // namespace VisualStudio
}  // namespace Microsoft
