// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/error_code_helper.h"

#include <cerrno>

#include "gtest/gtest.h"

#include "core/common/status.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace test {

namespace {
// Builds an Ort::Status from the given onnxruntime::Status with onnxruntime::ToOrtStatus().
Ort::Status AsOrtStatus(const Status& status) {
  Ort::Status ort_status{ToOrtStatus(status)};
  return ort_status;
}
}  // namespace

// A SYSTEM-category status carries a raw OS errno that must be mapped to a meaningful OrtErrorCode
// rather than being reinterpreted directly.
TEST(ToOrtStatusTest, SystemCategoryErrnoIsMapped) {
  EXPECT_EQ(AsOrtStatus(Status(common::SYSTEM, ENOENT, "no such file")).GetErrorCode(), ORT_NO_SUCHFILE);
  EXPECT_EQ(AsOrtStatus(Status(common::SYSTEM, EINVAL, "invalid")).GetErrorCode(), ORT_INVALID_ARGUMENT);
}

// An unrecognized errno falls back to ORT_FAIL instead of producing a bogus / out-of-range code.
TEST(ToOrtStatusTest, SystemCategoryUnknownErrnoFallsBackToFail) {
  EXPECT_EQ(AsOrtStatus(Status(common::SYSTEM, EACCES, "permission denied")).GetErrorCode(), ORT_FAIL);
}

// ONNXRUNTIME-category codes map 1:1 onto OrtErrorCode and must be preserved.
TEST(ToOrtStatusTest, OnnxRuntimeCategoryCodeIsPreserved) {
  EXPECT_EQ(AsOrtStatus(Status(common::ONNXRUNTIME, common::NO_SUCHFILE, "missing")).GetErrorCode(), ORT_NO_SUCHFILE);
  EXPECT_EQ(AsOrtStatus(Status(common::ONNXRUNTIME, common::INVALID_GRAPH, "bad graph")).GetErrorCode(),
            ORT_INVALID_GRAPH);
}

// The error message must be carried through unchanged.
TEST(ToOrtStatusTest, MessageIsPreserved) {
  auto ort_status = AsOrtStatus(Status(common::SYSTEM, ENOENT, "open file failed"));
  ASSERT_FALSE(ort_status.IsOK());
  EXPECT_EQ(ort_status.GetErrorMessage(), "open file failed");
}

}  // namespace test
}  // namespace onnxruntime
