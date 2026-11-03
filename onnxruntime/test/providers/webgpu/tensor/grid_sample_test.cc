// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <initializer_list>
#include <string>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
template <typename T>
class WebGpuGridSampleTest : public ::testing::Test {
};

using WebGpuGridSampleTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(WebGpuGridSampleTest, WebGpuGridSampleTestTypes);

TYPED_TEST(WebGpuGridSampleTest, test_grid_sample_16_4D_webgpu_empty_spatial_rejected) {
  // The WebGPU GridSample kernel is registered for opset 16-19 (NCHW, 4-D only) and now carries the
  // same non-empty spatial-dimension guard as the CPU and CUDA kernels. Exercise it directly on the
  // WebGPU provider with a zero-size spatial dimension; the guard rejects host-side before any shader
  // dispatch, so this validates the WebGPU path without depending on a particular GPU adapter.
  OpTester test("GridSample", 16);
  test.AddAttribute("mode", std::string("nearest"));
  test.AddAttribute("padding_mode", std::string("border"));
  test.AddAttribute("align_corners", int64_t{0});

  std::initializer_list<int64_t> X_shape{1, 1, 0, 5};
  std::initializer_list<TypeParam> X_data{};
  std::initializer_list<int64_t> Grid_shape{1, 2, 2, 2};
  std::initializer_list<TypeParam> Grid_data{
      TypeParam(0.0f), TypeParam(0.0f), TypeParam(-1.0f), TypeParam(-1.0f),
      TypeParam(1.0f), TypeParam(1.0f), TypeParam(0.5f), TypeParam(-0.5f)};
  std::initializer_list<int64_t> Y_shape{1, 1, 2, 2};

  test.AddInput<TypeParam>("X", X_shape, X_data);
  test.AddInput<TypeParam>("Grid", Grid_shape, Grid_data);
  test.AddOutput<TypeParam>("Y", Y_shape,
                            {TypeParam(0.0f), TypeParam(0.0f), TypeParam(0.0f), TypeParam(0.0f)});

  test.Config(OpTester::ExpectResult::kExpectFailure, "Input spatial dimensions must be non-empty for sampling")
      .ConfigEp(DefaultWebGpuExecutionProvider())
      .RunWithConfig();
}
#endif

}  // namespace test
}  // namespace onnxruntime
