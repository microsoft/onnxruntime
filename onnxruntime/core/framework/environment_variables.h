// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

// an enum encapsulating the different waiting thread policies
// For more details, please look up OMP_WAIT_POLICY under the version of the OpenMP spec that supports it - https://www.openmp.org/specifications/
enum OmpWaitPolicy {
  Active = 0,
  Passive
};

/**
  * Contains some ORT-specific environment variables that is set in the process running ORT
  */
struct EnvironmentVariables {
  int omp_num_threads;

  OmpWaitPolicy omp_wait_policy;
};
}  // namespace onnxruntime
