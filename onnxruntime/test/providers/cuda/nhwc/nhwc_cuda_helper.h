// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include <vector>
#include <utility>
#include <memory>

#include "gtest/gtest.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_provider_options.h"

#include "test/common/cuda_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/util/include/default_providers.h"

// extended cuda provider args. compare NHWC implementation vs CUDA NCHW and CPU EP.
#define MAKE_PROVIDERS_EPS_EXT(eps, pad_to_nc1d)                                \
  {                                                                             \
    std::vector<std::shared_ptr<IExecutionProvider>> execution_providers;       \
    OrtCUDAProviderOptionsV2 nhwc{};                                            \
    nhwc.prefer_nhwc = true;                                                    \
    nhwc.cudnn_conv1d_pad_to_nc1d = pad_to_nc1d;                                \
    execution_providers.push_back(CudaExecutionProviderWithOptions(&nhwc));     \
                                                                                \
    double error_tolerance = eps;                                               \
    OrtCUDAProviderOptionsV2 nchw{};                                            \
    nchw.prefer_nhwc = false;                                                   \
    nchw.cudnn_conv1d_pad_to_nc1d = pad_to_nc1d;                                \
    auto nchw_ep = CudaExecutionProviderWithOptions(&nchw);                     \
    auto test = op.get_test();                                                  \
    test->CompareEPs(std::move(nchw_ep), execution_providers, error_tolerance); \
    auto cpu_ep = DefaultCpuExecutionProvider();                                \
    test->CompareEPs(std::move(cpu_ep), execution_providers, error_tolerance);  \
  }

#define MAKE_PROVIDERS_EPS(eps) \
  MAKE_PROVIDERS_EPS_EXT(eps, false)

#define MAKE_PROVIDERS() MAKE_PROVIDERS_EPS(1e-3)

#define MAKE_PROVIDERS_EPS_TYPE_EXT(T, pad_to_nc1d) \
  if (std::is_same<T, MLFloat16>::value) {          \
    MAKE_PROVIDERS_EPS_EXT(2e-2, pad_to_nc1d)       \
  } else if (std::is_same<T, double>::value) {      \
    MAKE_PROVIDERS_EPS_EXT(2e-4, pad_to_nc1d)       \
  } else {                                          \
    MAKE_PROVIDERS_EPS_EXT(4e-3, pad_to_nc1d)       \
  }

#define MAKE_PROVIDERS_EPS_TYPE(T) \
  MAKE_PROVIDERS_EPS_TYPE_EXT(T, false)

namespace onnxruntime {
namespace test {

template <typename T>
class CudaNhwcTypedTest : public ::testing::Test {};

using CudaNhwcTestTypes = ::testing::Types<float, MLFloat16>;  // double,
TYPED_TEST_SUITE(CudaNhwcTypedTest, CudaNhwcTestTypes);
}  // namespace test
}  // namespace onnxruntime
