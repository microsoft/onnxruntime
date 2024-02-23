// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <unordered_map>
#include <gtest/gtest.h>

#include "runtimeParameters.h"

namespace RuntimeParameters {
std::unordered_map<std::string, std::string> Parameters;
}

namespace {
void usage(char** argv, int failedArgument) {
  std::cerr << "Unrecognized argument: " << argv[failedArgument] << "\n"
            << "Usage:\n\t" << argv[0] << " [/p:parameterName=parameterValue ...]\n";
}

bool parseArgument(const std::string& argument) {
  if (argument.rfind("/p:", 0) == 0) {
    // Parse argument in the form of /p:parameterName=parameterValue
    auto separatorIndex = argument.find('=');
    if (separatorIndex == std::string::npos || separatorIndex == 3) {
      return false;
    }
    auto parameterName = argument.substr(3, separatorIndex - 3);
    auto parameterValue = argument.substr(separatorIndex + 1);
    RuntimeParameters::Parameters[parameterName] = parameterValue;
    return true;
  }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  for (int i = 1; i < argc; i++) {
    if (!parseArgument(argv[i])) {
      usage(argv, i);
      return -1;
    }
  }
  return RUN_ALL_TESTS();
}
