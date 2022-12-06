// Copyright (c) Microsoft Corporation, Dale Phurrough. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/common.h"

#include <cstdio>

#include "gtest/gtest.h"

#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

class ConditionalOrt {
  Ort::Env ort_env{nullptr};

public:
  ConditionalOrt() {
    // create condition that will not be optimized-out by compiler to
    // simulate consumers of Ort conditionally and at a time of their
    // choosing the initialization of Ort API
    // e.g. checking if dependent data files or delay-loaded DLLs exist
    // before initializing or calling Ort APIs which require such dependencies
    char *const tmpname = std::tmpnam(nullptr);

    // condition test; intentionally failing to create test scenario
    if (!tmpname) {
      ort_env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "mylogid"};
      ORT_THROW("std::tmpnam() failed");
    }
  }
};

TEST(PlatformGetApiTest, Null_Env) {
#ifdef ORT_NO_EXCEPTIONS
  ConditionalOrt my_onnx;
#else
  EXPECT_NO_THROW({
    ConditionalOrt my_onnx;
  });
#endif
}

TEST(PlatformGetApiTest, Null_Session) {
  Ort::Session session{nullptr};
}

}  // namespace test
}  // namespace onnxruntime
