// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "gtest/gtest.h"

#include "test/test_environment.h"

namespace onnxruntime {
namespace test {

class GraphTransformationTests : public ::testing::Test {
 protected:
  GraphTransformationTests() {
    logger_ = DefaultLoggingManager().CreateLogger("GraphTransformationTests");
  }

  std::unique_ptr<logging::Logger> logger_;
};

}  // namespace test
}  // namespace onnxruntime
