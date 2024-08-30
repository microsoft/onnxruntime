// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "core/providers/shared_library/provider_api.h"  // Include this otherwise Windows build complains Env::Default() missing
#include "core/platform/env_var_utils.h"
#include "transformer_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// The environment variable is for testing purpose only, and it might be removed in the future.
// If you need some option in production, please file a feature request.
constexpr const char* kTransformerOptions = "ORT_TRANSFORMER_OPTIONS";

// Initialize the singleton instance
TransformerOptions TransformerOptions::instance;

const TransformerOptions* TransformerOptions::GetInstance() {
  if (!instance.initialized_) {
    // We do not use critical section here since it is fine to initialize multiple times by different threads.
    int value = ParseEnvironmentVariableWithDefault<int>(kTransformerOptions, 0);
    instance.Initialize(value);

    if (value > 0)
      std::cout << "ORT_TRANSFORMER_OPTIONS: IsPrecisionMode=" << instance.IsPrecisionMode()
                << ",DisablePersistentSoftmax=" << instance.DisablePersistentSoftmax()
                << ",DisableHalf2=" << instance.DisableHalf2()
                << std::endl;
  }

  return &instance;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
