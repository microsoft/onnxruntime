// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "core/session/onnxruntime_cxx_api.h"

int main(int argc, char** argv) {
  int status = 0;
   char cwd[PATH_MAX];
   if (getcwd(cwd, sizeof(cwd)) != NULL) {
       printf("Current working dir: %s\n", cwd);
   } else {
       perror("getcwd() error");
       return 1;
   }

  ORT_TRY {
    printf("")

    onnxruntime::test::TestEnvironment test_environment{argc, argv};

    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

  return status;
}
