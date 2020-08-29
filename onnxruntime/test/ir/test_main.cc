// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"

GTEST_API_ int main(int argc, char** argv) {
  int status = 0;

  ORT_TRY {
    onnxruntime::test::TestEnvironment test_environment{argc, argv};

    // Register Microsoft domain with min/max op_set version as 1/1.
    ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime_ir::kMSDomain, 1, 1);

    // Register Microsoft domain ops.
    onnxruntime_ir::MsOpRegistry::RegisterMsOps();

    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&ex]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

  return status;
}
