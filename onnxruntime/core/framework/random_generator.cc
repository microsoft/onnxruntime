// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_generator.h"
#include "core/framework/random_seed.h"

namespace onnxruntime {

RandomGenerator& RandomGenerator::Default() {
  static RandomGenerator generator(utils::GetRandomSeed());
  return generator;
}

PhiloxGenerator& PhiloxGenerator::Default() {
  static PhiloxGenerator generator(static_cast<uint64_t>(utils::GetRandomSeed()));
  return generator;
}

}  // namespace onnxruntime
