// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include <vector>
#include <utility>
#include <memory>

#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/common.h"

#include "test/providers/compare_provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

#include "gtest/gtest.h"

#define MAKE_PROVIDERS_EPS(eps)                                           \
  std::vector<std::shared_ptr<IExecutionProvider>> execution_providers;   \
  OrtCUDAProviderOptionsV2 nhwc{};                                        \
  nhwc.prefer_nhwc = true;                                                \
  execution_providers.push_back(CudaExecutionProviderWithOptions(&nhwc)); \
                                                                          \
  double error_tolerance = eps;                                           \
  OrtCUDAProviderOptionsV2 nchw{};                                        \
  nchw.prefer_nhwc = false;                                               \
  auto source_ep = CudaExecutionProviderWithOptions(&nchw);               \
  auto test = op.get_test();                                              \
  test->CompareEPs(std::move(source_ep), execution_providers, error_tolerance);

#define MAKE_PROVIDERS() MAKE_PROVIDERS_EPS(1e-3)

#define MAKE_PROVIDERS_EPS_TYPE(T)             \
  if (std::is_same<T, MLFloat16>::value) {     \
    MAKE_PROVIDERS_EPS(2e-2)                   \
  } else if (std::is_same<T, double>::value) { \
    MAKE_PROVIDERS_EPS(2e-4)                   \
  } else {                                     \
    MAKE_PROVIDERS_EPS(2e-3)                   \
  }
namespace onnxruntime {
namespace test {

template <typename T>
class CudaNhwcTypedTest : public ::testing::Test {};

using CudaNhwcTestTypes = ::testing::Types<float, MLFloat16>;  // double,
TYPED_TEST_SUITE(CudaNhwcTypedTest, CudaNhwcTestTypes);
}  // namespace test
}  // namespace onnxruntime
