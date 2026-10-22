// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "contrib_ops/cuda/llm/speculative_fold.h"

namespace onnxruntime {
namespace test {
namespace {

void CheckCuda(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t size) {
    CheckCuda(cudaMalloc(&data_, size));
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  template <typename T>
  T* As() const {
    return reinterpret_cast<T*>(data_);
  }
  void Upload(const void* source, size_t size) {
    CheckCuda(cudaMemcpy(data_, source, size, cudaMemcpyHostToDevice));
  }
  void Download(void* destination, size_t size) const {
    CheckCuda(cudaMemcpy(destination, data_, size, cudaMemcpyDeviceToHost));
  }

 private:
  void* data_ = nullptr;
};

class MappedRing {
 public:
  explicit MappedRing(size_t element_count) : element_count_(element_count) {
    CheckCuda(cudaHostAlloc(reinterpret_cast<void**>(&host_data_),
                            element_count * sizeof(int64_t),
                            cudaHostAllocMapped));
    CheckCuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_data_),
                                       host_data_, 0));
    std::fill(host_data_, host_data_ + element_count_, int64_t{0});
  }
  ~MappedRing() {
    if (host_data_ != nullptr) {
      cudaFreeHost(host_data_);
    }
  }
  MappedRing(const MappedRing&) = delete;
  MappedRing& operator=(const MappedRing&) = delete;

  int64_t* DeviceData() const { return device_data_; }
  int64_t* HostData() const { return host_data_; }

 private:
  size_t element_count_;
  int64_t* host_data_ = nullptr;
  int64_t* device_data_ = nullptr;
};

std::vector<__nv_bfloat16> MakeCache(int positions, float base) {
  std::vector<__nv_bfloat16> cache(static_cast<size_t>(positions) * 32);
  for (int position = 0; position < positions; ++position) {
    for (int feature = 0; feature < 32; ++feature) {
      cache[static_cast<size_t>(position) * 32 + feature] =
          __float2bfloat16(base + static_cast<float>(position * 32 + feature));
    }
  }
  return cache;
}

TEST(CUDA_EP_Unittest, FoldMid2AcceptCommitsStateAndPublishesRingRecord) {
  using onnxruntime::contrib::cuda::speculative_fold::LaunchFoldMid2;

  constexpr int64_t kRingSize = 4;
  const std::array<int64_t, 2> next_tokens = {7, 8};
  const std::array<int64_t, 2> input_ids = {0, 7};
  const std::array<int32_t, 2> status = {0, 4};
  const std::array<int64_t, 8> ctl = {0, 10, 99, 0, 2, 0, 0, 5};
  const std::array<int64_t, 2> ctx_commit = {1, 2};
  const std::array<int64_t, 2> ctx_a = {11, 12};
  const std::array<int64_t, 2> ctx_b = {21, 22};
  MappedRing ring(kRingSize * 8);
  const std::array<int64_t, 2> ptrs = {
      static_cast<int64_t>(reinterpret_cast<intptr_t>(ring.DeviceData())),
      kRingSize};
  const auto cos_cache = MakeCache(16, 1000.0f);
  const auto sin_cache = MakeCache(16, 2000.0f);

  DeviceBuffer d_next_tokens(sizeof(next_tokens));
  DeviceBuffer d_input_ids(sizeof(input_ids));
  DeviceBuffer d_status(sizeof(status));
  DeviceBuffer d_ctl(sizeof(ctl));
  DeviceBuffer d_ctx_commit(sizeof(ctx_commit));
  DeviceBuffer d_ctx_a(sizeof(ctx_a));
  DeviceBuffer d_ctx_b(sizeof(ctx_b));
  DeviceBuffer d_ptrs(sizeof(ptrs));
  DeviceBuffer d_cos_cache(cos_cache.size() * sizeof(__nv_bfloat16));
  DeviceBuffer d_sin_cache(sin_cache.size() * sizeof(__nv_bfloat16));
  DeviceBuffer d_head_ids(2 * sizeof(int64_t));
  DeviceBuffer d_head_c(sizeof(int32_t));
  DeviceBuffer d_slot_ids(2 * sizeof(int64_t));
  DeviceBuffer d_seqlens(sizeof(int32_t));
  DeviceBuffer d_cos_out(2 * 32 * sizeof(__nv_bfloat16));
  DeviceBuffer d_sin_out(2 * 32 * sizeof(__nv_bfloat16));
  DeviceBuffer d_sel(sizeof(int64_t));

  d_next_tokens.Upload(next_tokens.data(), sizeof(next_tokens));
  d_input_ids.Upload(input_ids.data(), sizeof(input_ids));
  d_status.Upload(status.data(), sizeof(status));
  d_ctl.Upload(ctl.data(), sizeof(ctl));
  d_ctx_commit.Upload(ctx_commit.data(), sizeof(ctx_commit));
  d_ctx_a.Upload(ctx_a.data(), sizeof(ctx_a));
  d_ctx_b.Upload(ctx_b.data(), sizeof(ctx_b));
  d_ptrs.Upload(ptrs.data(), sizeof(ptrs));
  d_cos_cache.Upload(cos_cache.data(), cos_cache.size() * sizeof(__nv_bfloat16));
  d_sin_cache.Upload(sin_cache.data(), sin_cache.size() * sizeof(__nv_bfloat16));

  const Status launch_status = LaunchFoldMid2(
      d_next_tokens.As<int64_t>(), d_input_ids.As<int64_t>(),
      d_status.As<int32_t>(), static_cast<int>(status.size()),
      d_ctl.As<int64_t>(), d_ctx_commit.As<int64_t>(),
      d_ctx_a.As<int64_t>(), d_ctx_b.As<int64_t>(), d_ptrs.As<int64_t>(),
      d_cos_cache.As<__nv_bfloat16>(), d_sin_cache.As<__nv_bfloat16>(),
      d_head_ids.As<int64_t>(), d_head_c.As<int32_t>(),
      d_slot_ids.As<int64_t>(), d_seqlens.As<int32_t>(),
      d_cos_out.As<__nv_bfloat16>(), d_sin_out.As<__nv_bfloat16>(),
      d_sel.As<int64_t>(), d_ctl.As<int64_t>(), d_ctx_commit.As<int64_t>(), 16, nullptr);
  ASSERT_TRUE(launch_status.IsOK()) << launch_status.ErrorMessage();
  CheckCuda(cudaStreamSynchronize(nullptr));

  std::array<int64_t, 8> actual_ctl{};
  std::array<int64_t, 2> actual_ctx_commit{};
  std::array<int64_t, 2> actual_head_ids{};
  std::array<int64_t, 2> actual_slot_ids{};
  std::array<int64_t, 1> actual_sel{};
  std::array<int32_t, 1> actual_head_c{};
  std::array<int32_t, 1> actual_seqlens{};
  std::array<__nv_bfloat16, 64> actual_cos{};
  std::array<__nv_bfloat16, 64> actual_sin{};
  d_ctl.Download(actual_ctl.data(), sizeof(actual_ctl));
  d_ctx_commit.Download(actual_ctx_commit.data(), sizeof(actual_ctx_commit));
  d_head_ids.Download(actual_head_ids.data(), sizeof(actual_head_ids));
  d_head_c.Download(actual_head_c.data(), sizeof(actual_head_c));
  d_slot_ids.Download(actual_slot_ids.data(), sizeof(actual_slot_ids));
  d_seqlens.Download(actual_seqlens.data(), sizeof(actual_seqlens));
  d_cos_out.Download(actual_cos.data(), sizeof(actual_cos));
  d_sin_out.Download(actual_sin.data(), sizeof(actual_sin));
  d_sel.Download(actual_sel.data(), sizeof(actual_sel));

  EXPECT_EQ(actual_head_ids, (std::array<int64_t, 2>{7, 8}));
  EXPECT_EQ(actual_head_c[0], 5);
  EXPECT_EQ(actual_slot_ids, (std::array<int64_t, 2>{5, 6}));
  EXPECT_EQ(actual_seqlens[0], 6);
  EXPECT_EQ(actual_sel[0], 1);
  EXPECT_EQ(actual_ctx_commit, (std::array<int64_t, 2>{21, 22}));
  EXPECT_EQ(actual_ctl[0], 8);
  EXPECT_EQ(actual_ctl[1], 12);
  EXPECT_EQ(actual_ctl[4], 3);
  EXPECT_EQ(actual_ctl[5], 2);
  EXPECT_EQ(actual_ctl[6], 1);
  EXPECT_EQ(actual_ctl[7], 7);

  for (int row = 0; row < 2; ++row) {
    for (int feature = 0; feature < 32; ++feature) {
      const size_t index = static_cast<size_t>(row) * 32 + feature;
      EXPECT_EQ(__bfloat162float(actual_cos[index]),
                __bfloat162float(cos_cache[static_cast<size_t>(11 + row) * 32 + feature]));
      EXPECT_EQ(__bfloat162float(actual_sin[index]),
                __bfloat162float(sin_cache[static_cast<size_t>(11 + row) * 32 + feature]));
    }
  }

  const int64_t* record = ring.HostData() + 8 * 2;
  EXPECT_EQ(record[0], 3);
  EXPECT_EQ(record[1], 7);
  EXPECT_EQ(record[2], 8);
  EXPECT_EQ(record[3], 1);
  EXPECT_EQ(record[4], 4);
  EXPECT_EQ(record[5], 8);
  EXPECT_EQ(record[6], 12);
}

TEST(CUDA_EP_Unittest, FoldMid2MutedRejectDoesNotCommitState) {
  using onnxruntime::contrib::cuda::speculative_fold::LaunchFoldMid2;

  const std::array<int64_t, 2> next_tokens = {5, 6};
  const std::array<int64_t, 2> input_ids = {0, 7};
  const std::array<int32_t, 1> status = {0};
  const std::array<int64_t, 8> ctl = {0, 4, 9, 1, 1, 0, 0, 2};
  const std::array<int64_t, 2> ctx_commit = {31, 32};
  const std::array<int64_t, 2> ctx_a = {41, 42};
  const std::array<int64_t, 2> ctx_b = {51, 52};
  MappedRing ring(8 * 8);
  std::fill(ring.HostData(), ring.HostData() + 8 * 8, int64_t{-1});
  const std::array<int64_t, 2> ptrs = {
      static_cast<int64_t>(reinterpret_cast<intptr_t>(ring.DeviceData())), 8};
  const auto cos_cache = MakeCache(16, 1000.0f);
  const auto sin_cache = MakeCache(16, 2000.0f);

  DeviceBuffer d_next_tokens(sizeof(next_tokens));
  DeviceBuffer d_input_ids(sizeof(input_ids));
  DeviceBuffer d_status(sizeof(status));
  DeviceBuffer d_ctl(sizeof(ctl));
  DeviceBuffer d_ctx_commit(sizeof(ctx_commit));
  DeviceBuffer d_ctx_a(sizeof(ctx_a));
  DeviceBuffer d_ctx_b(sizeof(ctx_b));
  DeviceBuffer d_ptrs(sizeof(ptrs));
  DeviceBuffer d_cos_cache(cos_cache.size() * sizeof(__nv_bfloat16));
  DeviceBuffer d_sin_cache(sin_cache.size() * sizeof(__nv_bfloat16));
  DeviceBuffer d_head_ids(2 * sizeof(int64_t));
  DeviceBuffer d_head_c(sizeof(int32_t));
  DeviceBuffer d_slot_ids(2 * sizeof(int64_t));
  DeviceBuffer d_seqlens(sizeof(int32_t));
  DeviceBuffer d_cos_out(2 * 32 * sizeof(__nv_bfloat16));
  DeviceBuffer d_sin_out(2 * 32 * sizeof(__nv_bfloat16));
  DeviceBuffer d_sel(sizeof(int64_t));

  d_next_tokens.Upload(next_tokens.data(), sizeof(next_tokens));
  d_input_ids.Upload(input_ids.data(), sizeof(input_ids));
  d_status.Upload(status.data(), sizeof(status));
  d_ctl.Upload(ctl.data(), sizeof(ctl));
  d_ctx_commit.Upload(ctx_commit.data(), sizeof(ctx_commit));
  d_ctx_a.Upload(ctx_a.data(), sizeof(ctx_a));
  d_ctx_b.Upload(ctx_b.data(), sizeof(ctx_b));
  d_ptrs.Upload(ptrs.data(), sizeof(ptrs));
  d_cos_cache.Upload(cos_cache.data(), cos_cache.size() * sizeof(__nv_bfloat16));
  d_sin_cache.Upload(sin_cache.data(), sin_cache.size() * sizeof(__nv_bfloat16));

  const Status launch_status = LaunchFoldMid2(
      d_next_tokens.As<int64_t>(), d_input_ids.As<int64_t>(),
      d_status.As<int32_t>(), static_cast<int>(status.size()),
      d_ctl.As<int64_t>(), d_ctx_commit.As<int64_t>(),
      d_ctx_a.As<int64_t>(), d_ctx_b.As<int64_t>(), d_ptrs.As<int64_t>(),
      d_cos_cache.As<__nv_bfloat16>(), d_sin_cache.As<__nv_bfloat16>(),
      d_head_ids.As<int64_t>(), d_head_c.As<int32_t>(),
      d_slot_ids.As<int64_t>(), d_seqlens.As<int32_t>(),
      d_cos_out.As<__nv_bfloat16>(), d_sin_out.As<__nv_bfloat16>(),
      d_sel.As<int64_t>(), d_ctl.As<int64_t>(), d_ctx_commit.As<int64_t>(), 16, nullptr);
  ASSERT_TRUE(launch_status.IsOK()) << launch_status.ErrorMessage();
  CheckCuda(cudaStreamSynchronize(nullptr));

  std::array<int64_t, 8> actual_ctl{};
  std::array<int64_t, 2> actual_ctx_commit{};
  std::array<int64_t, 2> actual_head_ids{};
  std::array<int64_t, 2> actual_slot_ids{};
  std::array<int64_t, 1> actual_sel{};
  d_ctl.Download(actual_ctl.data(), sizeof(actual_ctl));
  d_ctx_commit.Download(actual_ctx_commit.data(), sizeof(actual_ctx_commit));
  d_head_ids.Download(actual_head_ids.data(), sizeof(actual_head_ids));
  d_slot_ids.Download(actual_slot_ids.data(), sizeof(actual_slot_ids));
  d_sel.Download(actual_sel.data(), sizeof(actual_sel));

  EXPECT_EQ(actual_ctl, ctl);
  EXPECT_EQ(actual_ctx_commit, ctx_commit);
  EXPECT_EQ(actual_head_ids, (std::array<int64_t, 2>{5, 6}));
  EXPECT_EQ(actual_slot_ids, (std::array<int64_t, 2>{2, 3}));
  EXPECT_EQ(actual_sel[0], 0);
  EXPECT_TRUE(std::all_of(ring.HostData(), ring.HostData() + 8 * 8,
                          [](int64_t value) { return value == -1; }));
}

TEST(CUDA_EP_Unittest, FoldTailUpdatesDraftAndHonorsMute) {
  using onnxruntime::contrib::cuda::speculative_fold::LaunchFoldTail;

  const int64_t next_token = 42;
  const std::array<int32_t, 2> status = {0, 4};
  std::array<int64_t, 8> ctl = {0, 10, 9, 0, 3, 1, 0, 5};
  DeviceBuffer d_next_token(sizeof(next_token));
  DeviceBuffer d_status(sizeof(status));
  DeviceBuffer d_ctl(sizeof(ctl));
  DeviceBuffer d_done(sizeof(int32_t));
  d_next_token.Upload(&next_token, sizeof(next_token));
  d_status.Upload(status.data(), sizeof(status));
  d_ctl.Upload(ctl.data(), sizeof(ctl));

  Status launch_status = LaunchFoldTail(
      d_next_token.As<int64_t>(), d_status.As<int32_t>(),
      static_cast<int>(status.size()), d_ctl.As<int64_t>(),
      d_done.As<int32_t>(), d_ctl.As<int64_t>(), nullptr);
  ASSERT_TRUE(launch_status.IsOK()) << launch_status.ErrorMessage();
  CheckCuda(cudaStreamSynchronize(nullptr));

  std::array<int64_t, 8> actual_ctl{};
  int32_t done = 0;
  d_ctl.Download(actual_ctl.data(), sizeof(actual_ctl));
  d_done.Download(&done, sizeof(done));
  EXPECT_EQ(done, 1);
  EXPECT_EQ(actual_ctl[2], 42);
  EXPECT_EQ(actual_ctl[5], 3);

  ctl[2] = 17;
  ctl[3] = 1;
  d_ctl.Upload(ctl.data(), sizeof(ctl));
  launch_status = LaunchFoldTail(
      d_next_token.As<int64_t>(), d_status.As<int32_t>(),
      static_cast<int>(status.size()), d_ctl.As<int64_t>(),
      d_done.As<int32_t>(), d_ctl.As<int64_t>(), nullptr);
  ASSERT_TRUE(launch_status.IsOK()) << launch_status.ErrorMessage();
  CheckCuda(cudaStreamSynchronize(nullptr));
  d_ctl.Download(actual_ctl.data(), sizeof(actual_ctl));
  d_done.Download(&done, sizeof(done));
  EXPECT_EQ(done, 0);
  EXPECT_EQ(actual_ctl, ctl);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
