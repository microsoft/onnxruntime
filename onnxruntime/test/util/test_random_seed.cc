// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/test_random_seed.h"

#include <chrono>
#include <iostream>
#include <sstream>

#include "core/platform/env.h"

namespace onnxruntime {
namespace test {

namespace {
RandomSeedType LoadRandomSeed() {
  // parse from environment variable
  {
    const std::string value_str = Env::Default().GetEnvironmentVar(
        GetTestRandomSeedEnvironmentVariableName());

    if (!value_str.empty()) {
      std::istringstream is{value_str};
      RandomSeedType parsed_value;
      if (is >> std::noskipws >> parsed_value && is.eof()) {
        return parsed_value;
      } else {
        std::cerr << GetTestRandomSeedEnvironmentVariableName()
                  << " was set but not able to be parsed: \""
                  << value_str << "\"\n";
      }
    }
  }

  // generate from time
  return static_cast<RandomSeedType>(
      std::chrono::steady_clock::now().time_since_epoch().count());
}
}  // namespace

RandomSeedType GetTestRandomSeed() {
  static const RandomSeedType test_random_seed = LoadRandomSeed();
  return test_random_seed;
}

}  // namespace test
}  // namespace onnxruntime
