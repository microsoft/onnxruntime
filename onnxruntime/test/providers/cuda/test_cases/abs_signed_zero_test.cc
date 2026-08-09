// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cmath>

#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
namespace test {

TEST(CudaUnaryElementwiseTest, AbsReturnsPositiveZero) {
  const float input[] = {0.0f, -0.0f};
  float* device_input = nullptr;
  float* device_output = nullptr;

  ASSERT_TRUE(CUDA_CALL(cudaMalloc(&device_input, sizeof(input))).IsOK());
  ASSERT_TRUE(CUDA_CALL(cudaMalloc(&device_output, sizeof(input))).IsOK());
  ASSERT_TRUE(CUDA_CALL(cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice)).IsOK());

  Impl_Abs<float>(nullptr, device_input, device_output, 2);
  ASSERT_TRUE(CUDA_CALL(cudaDeviceSynchronize()).IsOK());

  float output[2]{};
  ASSERT_TRUE(CUDA_CALL(cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost)).IsOK());
  EXPECT_FALSE(std::signbit(output[0]));
  EXPECT_FALSE(std::signbit(output[1]));
  EXPECT_EQ(output[0], 0.0f);
  EXPECT_EQ(output[1], 0.0f);

  EXPECT_TRUE(CUDA_CALL(cudaFree(device_input)).IsOK());
  EXPECT_TRUE(CUDA_CALL(cudaFree(device_output)).IsOK());
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
