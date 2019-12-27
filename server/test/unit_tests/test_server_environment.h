// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "environment.h"

namespace onnxruntime {
namespace server {
namespace test {
ServerEnvironment* ServerEnv();
class TestServerEnvironment {
 public:
  TestServerEnvironment();
  ~TestServerEnvironment();

  TestServerEnvironment(const TestServerEnvironment&) = delete;
  TestServerEnvironment(TestServerEnvironment&&) = default;
};
}  // namespace test
}  // namespace server
}  // namespace onnxruntime
