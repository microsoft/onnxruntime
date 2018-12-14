// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include <gtest/gtest.h>

#ifdef _WIN32
typedef const wchar_t* PATH_TYPE;
#define TSTR(X) L##X
#else
#define TSTR(X) (X)
typedef const char* PATH_TYPE;
#endif

//empty
static inline void ORT_API_CALL MyLoggingFunction(void*, OrtLoggingLevel, const char*, const char*, const char*, const char*) {
}
template <bool use_customer_logger>
class CApiTestImpl : public ::testing::Test {
 protected:
  OrtEnv* env = nullptr;

  void SetUp() override {
    if (use_customer_logger) {
      ORT_THROW_ON_ERROR(OrtInitializeWithCustomLogger(MyLoggingFunction, nullptr, ONNXRUNTIME_LOGGING_LEVEL_kINFO, "Default", &env));
    } else {
      ORT_THROW_ON_ERROR(OrtInitialize(ONNXRUNTIME_LOGGING_LEVEL_kINFO, "Default", &env));
    }
  }

  void TearDown() override {
    if (env) OrtReleaseObject(env);
  }

  // Objects declared here can be used by all tests in the test case for Foo.
};

typedef CApiTestImpl<false> CApiTest;
