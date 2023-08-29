// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/common.h"

#include "test/providers/compare_provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

#include "gtest/gtest.h"

#define MAKE_PROVIDERS_EPS(eps)                                                  \
  std::vector<std::shared_ptr<IExecutionProvider>> execution_providers;   \
  OrtCUDAProviderOptionsV2 nhwc = {                                       \
      .prefer_nhwc = true};                                               \
  execution_providers.push_back(CudaExecutionProviderWithOptions(&nhwc)); \
                                                                          \
  double error_tolerance = eps;                                          \
  OrtCUDAProviderOptionsV2 nchw = {                                       \
      .prefer_nhwc = false};                                              \
  auto source_ep = CudaExecutionProviderWithOptions(&nchw);               \
  auto test = op.get_test();                                              \
  test->CompareEPs(std::move(source_ep), execution_providers, error_tolerance);

#define MAKE_PROVIDERS() MAKE_PROVIDERS_EPS(1e-3)
