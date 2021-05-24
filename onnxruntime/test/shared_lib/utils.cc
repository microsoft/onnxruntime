// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <limits>

OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream) {
  OrtCUDAProviderOptions cuda_options{
      0,
      OrtCudnnConvAlgoSearch::EXHAUSTIVE,
      std::numeric_limits<size_t>::max(),
      0,
      true,
      cuda_compute_stream != nullptr ? 1 : 0,
      cuda_compute_stream != nullptr ? cuda_compute_stream : nullptr,
      nullptr};
  return cuda_options;
}
