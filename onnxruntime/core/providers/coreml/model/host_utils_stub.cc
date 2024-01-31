// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>

#include "core/providers/coreml/model/host_utils.h"

namespace onnxruntime {
namespace coreml {
namespace util {

bool HasRequiredBaseOS() {
  return true;
}

int CoreMLVersion() {
  return 7;  // CoreML 7 is the latest we support
}

std::string GetTemporaryFilePath() {
  static std::atomic<int> counter = 0;

  // we want to avoid creating endless directories whilst avoiding clashes between tests running in parallel.
  return "coreml_model_packages/run." + std::to_string(counter++);
}

}  // namespace util
}  // namespace coreml
}  // namespace onnxruntime
