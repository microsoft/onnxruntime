// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"

#include "test_fixture.h"
using namespace onnxruntime;

TEST_F(CApiTest, session_options) {
  std::unique_ptr<ONNXRuntimeSessionOptions> options(ONNXRuntimeCreateSessionOptions());
  ASSERT_NE(options, nullptr);
}
