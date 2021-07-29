// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transformer_common.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// The environment variable is for testing purpose only, and it might be removed in the future.
// If you need some option in production, please file a feature request.
// The value is an integer, and its bits have the following meaning:
//    0x01 - precision mode.
constexpr const char* kTransformerOptions = "ORT_TRANSFORMER_OPTIONS";

// Initialize the singleton instance
TransformerOptions TransformerOptions::instance;

const TransformerOptions* TransformerOptions::GetInstance() {
  if (!instance.initialized_) {
    // We do not use critical section here since it is fine to initialize multiple times by different threads.
    int value = ParseEnvironmentVariableWithDefault<int>(kTransformerOptions, 0);
    instance.Initialize(value);
  }

  return &instance;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime