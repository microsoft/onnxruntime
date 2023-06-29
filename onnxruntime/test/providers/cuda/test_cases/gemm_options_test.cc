// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace cuda {
namespace test {

TEST(CudaGemmOptions, TestDefaultOptions) {
  HalfGemmOptions gemm_options;
  ASSERT_FALSE(gemm_options.IsCompute16F());
#if defined(USE_CUDA)
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_DEFAULT_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUBLAS_COMPUTE_32F);
#else
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_TENSOR_OP_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUDA_R_32F);
#endif
}

TEST(CudaGemmOptions, TestCompute16F) {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(1);
  ASSERT_TRUE(gemm_options.IsCompute16F());
#if defined(USE_CUDA)
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_DEFAULT_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUBLAS_COMPUTE_16F);
#else
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_TENSOR_OP_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUDA_R_16F);
#endif
}

TEST(CudaGemmOptions, NoReducedPrecision) {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(2);
  ASSERT_FALSE(gemm_options.IsCompute16F());
#if defined(USE_CUDA)
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  EXPECT_EQ(gemm_options.GetComputeType(), CUBLAS_COMPUTE_32F);
#else
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_TENSOR_OP_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUDA_R_32F);
#endif
}

TEST(CudaGemmOptions, Pedantic) {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(4);
  ASSERT_FALSE(gemm_options.IsCompute16F());
#if defined(USE_CUDA)
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_PEDANTIC_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUBLAS_COMPUTE_32F_PEDANTIC);
#else
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_TENSOR_OP_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUDA_R_32F);
#endif
}

TEST(CudaGemmOptions, Compute16F_Pedantic) {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(5);
  ASSERT_TRUE(gemm_options.IsCompute16F());
#if defined(USE_CUDA)
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_PEDANTIC_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUBLAS_COMPUTE_16F_PEDANTIC);
#else
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_TENSOR_OP_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUDA_R_16F);
#endif
}

TEST(CudaGemmOptions, Compute16F_NoReducedPrecision) {
  HalfGemmOptions gemm_options;
  gemm_options.Initialize(3);
  ASSERT_TRUE(gemm_options.IsCompute16F());
#if defined(USE_CUDA)
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_DEFAULT_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUBLAS_COMPUTE_16F);
#else
  EXPECT_EQ(gemm_options.GetMathMode(), CUBLAS_TENSOR_OP_MATH);
  EXPECT_EQ(gemm_options.GetComputeType(), CUDA_R_16F);
#endif
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
