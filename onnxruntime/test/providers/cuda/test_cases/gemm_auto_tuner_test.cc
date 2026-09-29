// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

#include "core/providers/cuda/math/gemm_auto_tuner.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace cuda {
namespace test {
namespace {

using OptionalString = std::optional<std::string>;

GemmTuneKey MakeTestKey(int m) {
  GemmTuneKey key;
  // Not a real device, so the entries never collide with kernels tuned by other tests.
  key.device_uuid.fill('t');
  key.data_type = GemmDataType::kBFloat16;
  key.m = m;
  key.n = 48;
  key.k = 5120;
  return key;
}

class CudaStreamGuard {
 public:
  CudaStreamGuard() { CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~CudaStreamGuard() { cudaStreamDestroy(stream_); }
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_{};
};

GemmTuneCandidate DelayCandidate(GemmKernel kernel, cudaStream_t stream, uint64_t nanoseconds) {
  return {kernel, [stream, nanoseconds]() { return LaunchGpuDelay(stream, nanoseconds); }};
}

TEST(GemmAutoTunerTest, DispatchPolicyDefaultsToCublas) {
  EXPECT_EQ(ResolveGemmDispatchPolicy(std::nullopt, std::nullopt, std::nullopt), GemmDispatchPolicy::kCublas);
  EXPECT_EQ(ResolveGemmDispatchPolicy(OptionalString{""}, OptionalString{" "}, OptionalString{""}),
            GemmDispatchPolicy::kCublas);
}

TEST(GemmAutoTunerTest, DispatchPolicyPrecedence) {
  // The session config enables or disables tuning and wins over the environment.
  EXPECT_EQ(ResolveGemmDispatchPolicy(std::nullopt, OptionalString{"1"}, std::nullopt), GemmDispatchPolicy::kAutoTune);
  EXPECT_EQ(ResolveGemmDispatchPolicy(std::nullopt, OptionalString{"0"}, OptionalString{"1"}),
            GemmDispatchPolicy::kCublas);
  EXPECT_EQ(ResolveGemmDispatchPolicy(std::nullopt, std::nullopt, OptionalString{"1"}), GemmDispatchPolicy::kAutoTune);
  EXPECT_EQ(ResolveGemmDispatchPolicy(std::nullopt, std::nullopt, OptionalString{"0"}), GemmDispatchPolicy::kCublas);

  // ORT_ENABLE_SMALL_N_GEMV forces a kernel whatever the tuning setting.
  EXPECT_EQ(ResolveGemmDispatchPolicy(OptionalString{"1"}, std::nullopt, std::nullopt),
            GemmDispatchPolicy::kSmallNGemv);
  EXPECT_EQ(ResolveGemmDispatchPolicy(OptionalString{"1"}, OptionalString{"0"}, std::nullopt),
            GemmDispatchPolicy::kSmallNGemv);
  EXPECT_EQ(ResolveGemmDispatchPolicy(OptionalString{"0"}, OptionalString{"1"}, OptionalString{"1"}),
            GemmDispatchPolicy::kCublas);
}

TEST(GemmAutoTunerTest, ParsesOnOffValues) {
  for (const char* value : {"1", "true", "ON", " True "}) {
    EXPECT_EQ(ParseGemmOnOffOption(OptionalString{value}, "test"), std::optional<bool>{true}) << value;
  }
  for (const char* value : {"0", "false", "Off"}) {
    EXPECT_EQ(ParseGemmOnOffOption(OptionalString{value}, "test"), std::optional<bool>{false}) << value;
  }
  EXPECT_FALSE(ParseGemmOnOffOption(std::nullopt, "test").has_value());
  EXPECT_ANY_THROW(ParseGemmOnOffOption(OptionalString{"2"}, "test"));
  EXPECT_ANY_THROW(ParseGemmOnOffOption(OptionalString{"yes please"}, "test"));
}

TEST(GemmAutoTunerTest, PickFastestKeepsDefaultWithinMargin) {
  EXPECT_EQ(PickFastestGemmCandidate({1.0f}, 1.05f), 0u);
  EXPECT_EQ(PickFastestGemmCandidate({1.0f, 0.5f}, 1.05f), 1u);
  EXPECT_EQ(PickFastestGemmCandidate({1.0f, 0.96f}, 1.05f), 0u);
  EXPECT_EQ(PickFastestGemmCandidate({1.0f, 1.0f}, 1.05f), 0u);
  EXPECT_EQ(PickFastestGemmCandidate({0.5f, 1.0f}, 1.05f), 0u);
  EXPECT_EQ(PickFastestGemmCandidate({1.0f, 0.8f, 0.5f}, 1.05f), 2u);
}

TEST(GemmAutoTunerTest, CacheKeepsFirstInsertion) {
  GemmAutoTuneCache& cache = GemmAutoTuneCache::Instance();
  const GemmTuneKey key = MakeTestKey(1001);
  EXPECT_FALSE(cache.Lookup(key).has_value());
  EXPECT_EQ(cache.Insert(key, GemmKernel::kSmallNGemv), GemmKernel::kSmallNGemv);
  EXPECT_EQ(cache.Insert(key, GemmKernel::kCublas), GemmKernel::kSmallNGemv);
  EXPECT_EQ(cache.Lookup(key), std::optional<GemmKernel>{GemmKernel::kSmallNGemv});

  GemmTuneKey other = key;
  other.small_n_vectorized = !key.small_n_vectorized;
  EXPECT_FALSE(cache.Lookup(other).has_value());
}

TEST(GemmAutoTunerTest, TimesAndSelectsFasterCandidate) {
  CudaStreamGuard stream;
  std::vector<float> times_ms;
  ASSERT_STATUS_OK(TimeGemmCandidates(stream.get(),
                                      {DelayCandidate(GemmKernel::kCublas, stream.get(), 200'000),
                                       DelayCandidate(GemmKernel::kSmallNGemv, stream.get(), 20'000)},
                                      GemmTuneL2State{}, times_ms));
  ASSERT_EQ(times_ms.size(), 2u);
  EXPECT_GT(times_ms[0], 0.15f);
  EXPECT_LT(times_ms[1], 0.15f);

  GemmKernel selected = GemmKernel::kCublas;
  const GemmTuneKey fast_key = MakeTestKey(1002);
  ASSERT_STATUS_OK(TuneGemmKernel(fast_key, stream.get(),
                                  {DelayCandidate(GemmKernel::kCublas, stream.get(), 200'000),
                                   DelayCandidate(GemmKernel::kSmallNGemv, stream.get(), 20'000)},
                                  GemmTuneL2State{}, selected));
  EXPECT_EQ(selected, GemmKernel::kSmallNGemv);
  EXPECT_EQ(GemmAutoTuneCache::Instance().Lookup(fast_key), std::optional<GemmKernel>{GemmKernel::kSmallNGemv});

  const GemmTuneKey slow_key = MakeTestKey(1003);
  ASSERT_STATUS_OK(TuneGemmKernel(slow_key, stream.get(),
                                  {DelayCandidate(GemmKernel::kCublas, stream.get(), 20'000),
                                   DelayCandidate(GemmKernel::kSmallNGemv, stream.get(), 200'000)},
                                  GemmTuneL2State{}, selected));
  EXPECT_EQ(selected, GemmKernel::kCublas);

  // A cached key is not re-timed, even if the candidates changed.
  ASSERT_STATUS_OK(TuneGemmKernel(slow_key, stream.get(),
                                  {DelayCandidate(GemmKernel::kCublas, stream.get(), 200'000),
                                   DelayCandidate(GemmKernel::kSmallNGemv, stream.get(), 20'000)},
                                  GemmTuneL2State{}, selected));
  EXPECT_EQ(selected, GemmKernel::kCublas);
}

TEST(GemmAutoTunerTest, FlushBufferIsSizedFromL2) {
  cudaDeviceProp prop{};
  prop.l2CacheSize = 0;
  EXPECT_EQ(GemmAutoTuneFlushBytes(prop), 0u);
  prop.l2CacheSize = 3 << 20;
  EXPECT_EQ(GemmAutoTuneFlushBytes(prop), size_t{6} << 20);
  prop.l2CacheSize = 1 << 30;
  EXPECT_EQ(GemmAutoTuneFlushBytes(prop), size_t{256} << 20);

  CudaStreamGuard stream;
  void* buffer = nullptr;
  const size_t flush_bytes = size_t{1} << 20;
  CUDA_CALL_THROW(cudaMalloc(&buffer, flush_bytes + 64));
  GemmTuneL2State l2;
  l2.flush_buffer = buffer;
  l2.flush_bytes = flush_bytes;
  // An unaligned, odd-sized hot buffer is read in whole 16-byte words only.
  l2.hot_buffer = static_cast<char*>(buffer) + flush_bytes + 2;
  l2.hot_bytes = 45;
  std::vector<float> times_ms;
  const Status status =
      TimeGemmCandidates(stream.get(), {DelayCandidate(GemmKernel::kCublas, stream.get(), 1'000)}, l2, times_ms);
  cudaFree(buffer);
  ASSERT_STATUS_OK(status);
  EXPECT_EQ(times_ms.size(), 1u);
}

TEST(GemmAutoTunerTest, DetectsStreamCapture) {
  CudaStreamGuard stream;
  bool capturing = true;
  ASSERT_STATUS_OK(IsCudaStreamCapturing(stream.get(), capturing));
  EXPECT_FALSE(capturing);

  CUDA_CALL_THROW(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  const Status status = IsCudaStreamCapturing(stream.get(), capturing);
  cudaGraph_t graph{};
  CUDA_CALL_THROW(cudaStreamEndCapture(stream.get(), &graph));
  CUDA_CALL_THROW(cudaGraphDestroy(graph));
  ASSERT_STATUS_OK(status);
  EXPECT_TRUE(capturing);
}

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
