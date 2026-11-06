// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
#include <gsl/gsl>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {

TEST(MatMulNBitsGemv, SingleWarp4BitsEligibility) {
  using contrib::cuda::IsSupportedSingleWarp4BitsGemvShape;
  for (int n : {1024, 2048}) {
    EXPECT_TRUE(IsSupportedSingleWarp4BitsGemvShape(true, false, false, 1, n, 2048, 32));
  }
  EXPECT_TRUE(IsSupportedSingleWarp4BitsGemvShape(true, false, false, 1, 2048, 6144, 32));
  EXPECT_FALSE(IsSupportedSingleWarp4BitsGemvShape(false, false, false, 1, 2048, 2048, 32));
  EXPECT_FALSE(IsSupportedSingleWarp4BitsGemvShape(true, true, false, 1, 2048, 2048, 32));
  EXPECT_FALSE(IsSupportedSingleWarp4BitsGemvShape(true, false, true, 1, 2048, 2048, 32));
  for (int m : {0, 2, 8, 16, 17}) {
    EXPECT_FALSE(IsSupportedSingleWarp4BitsGemvShape(true, false, false, m, 2048, 2048, 32));
  }
  for (int block_size : {16, 64, 128}) {
    EXPECT_FALSE(IsSupportedSingleWarp4BitsGemvShape(true, false, false, 1, 2048, 2048, block_size));
  }
  for (int n : {32, 2052, 6144, 151936, 248320}) {
    EXPECT_FALSE(IsSupportedSingleWarp4BitsGemvShape(true, false, false, 1, n, 2048, 32));
  }
  EXPECT_FALSE(IsSupportedSingleWarp4BitsGemvShape(true, false, false, 1, 2048, 2056, 32));
}

TEST(MatMulNBitsGemv, ExistingSpecializationsRemainEligible) {
  using contrib::cuda::IsSupportedSingleRowGemvShape;
  EXPECT_TRUE(IsSupportedSingleRowGemvShape(false, 1, 32, 2880, 32));
  EXPECT_TRUE(IsSupportedSingleRowGemvShape(false, 1, 32, 2880, 64));
  EXPECT_TRUE(IsSupportedSingleRowGemvShape(false, 1, 248320, 2560, 32));
  EXPECT_FALSE(IsSupportedSingleRowGemvShape(true, 1, 32, 2880, 32));
  EXPECT_FALSE(IsSupportedSingleRowGemvShape(false, 2, 32, 2880, 32));
}

class MatMulNBitsGemvDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CALL_THROW(cudaSetDevice(0));
    CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CALL_THROW(cudaMalloc(&input_, 6144 * sizeof(half)));
    CUDA_CALL_THROW(cudaMalloc(&output_, 6144 * sizeof(half)));
    CUDA_CALL_THROW(cudaMalloc(&weights_, 6144 * 6144 / 2));
    CUDA_CALL_THROW(cudaMalloc(&scales_, 6144 * 6144 / 16 * sizeof(half)));
    CUDA_CALL_THROW(cudaMalloc(&zero_points_, 6144 * 6144 / 16));
    CUDA_CALL_THROW(cudaMemsetAsync(input_, 0, 6144 * sizeof(half), stream_));
    CUDA_CALL_THROW(cudaMemsetAsync(weights_, 0, 6144 * 6144 / 2, stream_));
    CUDA_CALL_THROW(cudaMemsetAsync(scales_, 0, 6144 * 6144 / 16 * sizeof(half), stream_));
    CUDA_CALL_THROW(cudaMemsetAsync(zero_points_, 0, 6144 * 6144 / 16, stream_));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    CUDA_CALL_THROW(cudaDeviceGetAttribute(&shared_memory_, cudaDevAttrMaxSharedMemoryPerBlock, 0));
  }

  void TearDown() override {
    EXPECT_EQ(cudaFree(zero_points_), cudaSuccess);
    EXPECT_EQ(cudaFree(scales_), cudaSuccess);
    EXPECT_EQ(cudaFree(weights_), cudaSuccess);
    EXPECT_EQ(cudaFree(output_), cudaSuccess);
    EXPECT_EQ(cudaFree(input_), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

  template <typename T = half>
  void CheckLaunch(int n, int k, bool single_warp, int block_size = 32,
                   bool has_zero_points = false, bool has_bias = false, bool invalid_control = false) {
    SCOPED_TRACE(MakeString("N=", n, ", K=", k, ", block_size=", block_size));
    cudaGraph_t graph = nullptr;
    bool capturing = false;
    const auto cleanup = gsl::finally([&] {
      if (capturing) {
        EXPECT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
      }
      if (graph != nullptr) {
        EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
      }
    });
    const auto launch = [&] {
      return contrib::cuda::TryMatMul4Bits<T>(
          reinterpret_cast<T*>(output_), reinterpret_cast<const T*>(input_), weights_,
          reinterpret_cast<const T*>(scales_), has_zero_points ? zero_points_ : nullptr,
          has_bias ? reinterpret_cast<const T*>(output_) : nullptr,
          1, n, k, block_size, static_cast<size_t>(shared_memory_), stream_);
    };
    // Keep lazy kernel loading outside capture, including on older CUDA runtimes.
    if (!invalid_control) {
      EXPECT_EQ(launch(), !has_bias);
      CUDA_CALL_THROW(cudaGetLastError());
      CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    }
    CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    capturing = true;
    if (invalid_control) {
      EXPECT_THROW(launch(), OnnxRuntimeException);
    } else {
      EXPECT_EQ(launch(), !has_bias);
    }
    CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &graph));
    capturing = false;
    size_t count = 0;
    CUDA_CALL_THROW(cudaGraphGetNodes(graph, nullptr, &count));
    if (has_bias || invalid_control) {
      EXPECT_EQ(count, 0u);
      return;
    }
    ASSERT_EQ(count, 1u);
    cudaGraphNode_t node = nullptr;
    CUDA_CALL_THROW(cudaGraphGetNodes(graph, &node, &count));
    cudaGraphNodeType type{};
    CUDA_CALL_THROW(cudaGraphNodeGetType(node, &type));
    ASSERT_EQ(type, cudaGraphNodeTypeKernel);
    cudaKernelNodeParams params{};
    CUDA_CALL_THROW(cudaGraphKernelNodeGetParams(node, &params));
    EXPECT_EQ(params.blockDim.x, 32u);
    EXPECT_EQ(params.blockDim.y, single_warp ? 1u : 8u);
    EXPECT_EQ(params.blockDim.z, 1u);
    EXPECT_EQ(params.gridDim.x, static_cast<unsigned int>(single_warp ? n : n / 8));
    EXPECT_EQ(params.gridDim.y, 1u);
    EXPECT_EQ(params.gridDim.z, 1u);
  }

  EnvVarMap Controls(optional<std::string> generic, optional<std::string> legacy,
                     optional<std::string> disable = {}) {
    return {{"ORT_ENABLE_SINGLE_WARP_INT4_GEMV", std::move(generic)},
            {"ORT_ENABLE_QWEN2B_GEMV_SPECIALIZATION", std::move(legacy)},
            {"ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION", std::move(disable)}};
  }

  cudaStream_t stream_ = nullptr;
  half *input_ = nullptr, *output_ = nullptr, *scales_ = nullptr;
  uint8_t *weights_ = nullptr, *zero_points_ = nullptr;
  int shared_memory_ = 0;
};

TEST_F(MatMulNBitsGemvDispatchTest, GenericOptInSelectsAllTunedKernels) {
  ScopedEnvironmentVariables controls{Controls("1", {})};
  CheckLaunch(1024, 2048, true);
  CheckLaunch(2048, 2048, true);
  CheckLaunch(2048, 6144, true);
}

TEST_F(MatMulNBitsGemvDispatchTest, LegacyAliasSelectsTunedKernel) {
  ScopedEnvironmentVariables controls{Controls({}, "1")};
  CheckLaunch(2048, 2048, true);
}

TEST_F(MatMulNBitsGemvDispatchTest, DefaultKeepsExistingKernel) {
  ScopedEnvironmentVariables controls{Controls({}, {})};
  CheckLaunch(2048, 2048, false);
}

TEST_F(MatMulNBitsGemvDispatchTest, GenericOptOutOverridesLegacyOptIn) {
  ScopedEnvironmentVariables controls{Controls("0", "1")};
  CheckLaunch(2048, 2048, false);
}

TEST_F(MatMulNBitsGemvDispatchTest, GenericOptInIgnoresLegacyOptOutAndMalformedValues) {
  for (const auto* legacy : {"0", "invalid"}) {
    SCOPED_TRACE(legacy);
    ScopedEnvironmentVariables controls{Controls("1", legacy)};
    CheckLaunch(2048, 2048, true);
  }
}

TEST_F(MatMulNBitsGemvDispatchTest, GlobalDisableOverridesBothOptIns) {
  ScopedEnvironmentVariables controls{Controls("1", "1", "1")};
  CheckLaunch(2048, 2048, false);
}

TEST_F(MatMulNBitsGemvDispatchTest, UnsupportedShapesAndMetadataKeepExistingKernels) {
  ScopedEnvironmentVariables controls{Controls("1", "1")};
  CheckLaunch(6144, 2048, false);
  CheckLaunch(2056, 2048, false);
  CheckLaunch(2048, 2048, false, 64);
  CheckLaunch(2048, 2048, false, 32, true);
  CheckLaunch<float>(2048, 2048, false);
}

TEST_F(MatMulNBitsGemvDispatchTest, BiasFallsBackWithoutLaunchingAGemvKernel) {
  ScopedEnvironmentVariables controls{Controls("1", "1")};
  CheckLaunch(2048, 2048, false, 32, false, true);
}

TEST_F(MatMulNBitsGemvDispatchTest, InvalidGenericControlIsNotHiddenByLegacyOptIn) {
  ScopedEnvironmentVariables controls{Controls("invalid", "1")};
  CheckLaunch(2048, 2048, false, 32, false, false, true);
}

}  // namespace test
}  // namespace onnxruntime
#endif
