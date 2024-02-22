// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>

#include "core/platform/env.h"
#include "core/providers/coreml/model/host_utils.h"

namespace onnxruntime {
namespace coreml {
namespace util {

bool HasRequiredBaseOS() {
  return true;
}

int CoreMLVersion() {
  return 7;  // CoreML 7 is the latest we support.
}

std::string GetTemporaryFilePath() {
  static std::atomic<int> counter = 0;

  // we want to avoid creating endless directories/names whilst avoiding clashes if tests run in parallel so cycle
  // through 20 potential output names.
  auto dir_name = "coreml_ep_test_run." + std::to_string(counter++ % 20);

  // to replicate the iOS/macOS host_utils.mm behavior where the output is <user temporary directory>/<unique_name>
  // we want to return the name of something that does not exist. this is required for ML Package creation.
  auto& env = Env::Default();
  if (env.FolderExists(dir_name)) {
    ORT_THROW_IF_ERROR(env.DeleteFolder(ToPathString(dir_name)));
  }

  return dir_name;
}

}  // namespace util
}  // namespace coreml
}  // namespace onnxruntime
