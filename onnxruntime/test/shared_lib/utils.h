// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_cxx_api.h"

inline OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream) {
  OrtCUDAProviderOptions cuda_options{
      0,
      OrtCudnnConvAlgoSearch::EXHAUSTIVE,
      std::numeric_limits<size_t>::max(),
      0,
      true,
      cuda_compute_stream != nullptr ? 1 : 0,
      cuda_compute_stream != nullptr ? cuda_compute_stream : nullptr};
  return cuda_options;
}
