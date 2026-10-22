// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "contrib_ops/cuda/llm/speculative_fold_v2.h"

namespace onnxruntime::test {
namespace {
namespace fold = contrib::cuda::speculative_fold_v2;

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
struct Buffer {
  explicit Buffer(size_t count) : count(count) { Check(cudaMalloc(&data, count * sizeof(T))); }
  ~Buffer() {
    if (cudaFree(data) != cudaSuccess) std::fprintf(stderr, "Fold test cudaFree failed\n");
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void Upload(const std::vector<T>& values) {
    ASSERT_EQ(values.size(), count);
    Check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> Read() const {
    std::vector<T> values(count);
    Check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
  T* data{};
  size_t count;
};

struct Fixture {
  Buffer<int64_t> tokens{8}, ids{8}, ctl{16}, context{2}, prefixes{8}, prefix2{2}, ptrs{2}, ring{16};
  Buffer<int32_t> status{2}, head_c{1}, lengths{1}, done{1};
  Buffer<int64_t> head_ids{4}, slots{4}, selection{1};
  Buffer<uint16_t> cosine{16 * 3}, sine{16 * 3}, head_cos{12}, head_sin{12}, hidden{24}, out_hidden{12};

  void Reset(int rows, int accepted, bool muted = false, int64_t sequence = 0) {
    std::vector<int64_t> predictions{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<int64_t> input(8, -1);
    input[0] = 99;
    for (int i = 0; i < accepted; ++i) input[i + 1] = predictions[i];
    tokens.Upload(predictions);
    ids.Upload(input);
    std::vector<int64_t> control(16, 0);
    control[1] = 3;
    control[3] = muted ? 1 : 0;
    control[4] = sequence;
    control[7] = 2;
    ctl.Upload(control);
    context.Upload({-1, -2});
    prefixes.Upload({100, 101, 200, 201, 300, 301, 400, 401});
    prefix2.Upload({200, 201});
    ptrs.Upload({reinterpret_cast<int64_t>(ring.data), 2});
    ring.Upload(std::vector<int64_t>(16, -7));
    status.Upload({0, 0});
    std::vector<uint16_t> cos(48), sin(48), h(24);
    for (size_t i = 0; i < cos.size(); ++i) {
      cos[i] = static_cast<uint16_t>(i + 1);
      sin[i] = static_cast<uint16_t>(i + 100);
    }
    for (size_t i = 0; i < h.size(); ++i) h[i] = static_cast<uint16_t>(i + 200);
    cosine.Upload(cos);
    sine.Upload(sin);
    hidden.Upload(h);
    (void)rows;
  }

  Status Mid(int rows, int heads, cudaStream_t stream = nullptr) {
    return fold::LaunchMid(tokens.data, ids.data, status.data, 2, ctl.data, context.data,
                           prefixes.data, prefix2.data, ptrs.data, cosine.data, sine.data,
                           head_ids.data, head_c.data, slots.data, lengths.data, head_cos.data, head_sin.data,
                           selection.data, hidden.data, out_hidden.data, 3, rows, heads, 16, 3, 16, stream);
  }
  Status Tail(int slot = 2, cudaStream_t stream = nullptr) {
    return fold::LaunchTail(tokens.data, status.data, 2, ctl.data, ptrs.data, done.data, slot, stream);
  }
};

class FoldContribV2Test : public testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "CUDA is required";
  }
};

TEST_F(FoldContribV2Test, AllPrefixLengthsAndDeferredRingPublication) {
  Fixture f;
  for (int rows : {1, 2, 3, 4}) {
    const int heads = std::max(2, rows);
    for (int accepted = 0; accepted < rows; ++accepted) {
      SCOPED_TRACE(testing::Message() << rows << "/" << accepted);
      f.Reset(rows, accepted, false, 3);
      ASSERT_TRUE(f.Mid(rows, heads).IsOK());
      Check(cudaDeviceSynchronize());
      EXPECT_EQ(f.selection.Read()[0], accepted);
      EXPECT_EQ(f.context.Read(), (std::vector<int64_t>{100 * (accepted + 1), 100 * (accepted + 1) + 1}));
      const auto ctl = f.ctl.Read();
      EXPECT_EQ(ctl[0], 10 * (accepted + 1));
      EXPECT_EQ(ctl[1], 4 + accepted);
      EXPECT_EQ(ctl[7], 3 + accepted);
      EXPECT_EQ(ctl[4], 4);
      EXPECT_EQ(ctl[6], accepted);
      auto record = f.ring.Read();
      EXPECT_EQ(record[8], -7);
      EXPECT_EQ(record[11], accepted);
      EXPECT_EQ(record[13], 10 * (accepted + 1));
      EXPECT_EQ(record[14], 4 + accepted);
      if (rows > 2) {
        const uint64_t packed = 30 | (rows > 3 ? uint64_t{40} << 32 : 0);
        EXPECT_EQ(static_cast<uint64_t>(record[15]), packed);
      }
      ASSERT_TRUE(f.Tail().IsOK());
      Check(cudaDeviceSynchronize());
      EXPECT_EQ(f.ring.Read()[8], 4);
      EXPECT_EQ(f.done.Read()[0], 0);
      const auto cos = f.head_cos.Read();
      EXPECT_EQ(cos[0], 13);
      EXPECT_EQ(cos[3 * heads - 1], 12 + 3 * heads);
      const auto hidden = f.out_hidden.Read();
      for (int row = 0; row < heads; ++row)
        for (int col = 0; col < 3; ++col)
          EXPECT_EQ(hidden[row * 3 + col], 200 + std::min(row, rows - 1) * 3 + col);
    }
  }
}

TEST_F(FoldContribV2Test, MutedPrefillDoesNotAccessRingOrCommitContext) {
  Fixture f;
  f.Reset(8, 0, true);
  f.ptrs.Upload({0, 0});
  ASSERT_TRUE(f.Mid(8, 4).IsOK());
  ASSERT_TRUE(f.Tail().IsOK());
  Check(cudaDeviceSynchronize());
  EXPECT_EQ(f.context.Read(), (std::vector<int64_t>{-1, -2}));
  EXPECT_EQ(f.ctl.Read()[1], 3);
  EXPECT_EQ(f.ctl.Read()[4], 0);
  EXPECT_EQ(f.head_c.Read()[0], 0);
  EXPECT_EQ(f.head_cos.Read()[0], 4);
  EXPECT_EQ(f.selection.Read()[0], 0);
}

TEST_F(FoldContribV2Test, InvalidDeviceMetadataMutesBeforeRingAccess) {
  Fixture f;
  for (int kind = 0; kind < 4; ++kind) {
    f.Reset(4, 3);
    auto ctl = f.ctl.Read();
    const int bit = kind < 2 ? 16 : (kind == 2 ? 8 : 4);
    if (kind == 0) f.ptrs.Upload({0, 0});
    if (kind == 1) ctl[4] = INT64_MAX;
    if (kind == 2) ctl[1] = INT64_MAX;
    if (kind == 3) ctl[7] = -1;
    f.ctl.Upload(ctl);
    ASSERT_TRUE(f.Mid(4, 4).IsOK());
    ASSERT_TRUE(f.Tail().IsOK());
    Check(cudaDeviceSynchronize());
    EXPECT_EQ(f.selection.Read()[0], -1);
    EXPECT_EQ(f.ctl.Read()[5] & bit, bit);
    EXPECT_EQ(f.ctl.Read()[3], 1);
    EXPECT_EQ(f.context.Read(), (std::vector<int64_t>{-1, -2}));
    EXPECT_EQ(f.ring.Read(), std::vector<int64_t>(16, -7));
  }
}

TEST_F(FoldContribV2Test, TailSlotsAndValidationArePublished) {
  Fixture f;
  for (int slot : {2, 8, 9}) {
    f.Reset(4, 2);
    f.status.Upload({1, 4});
    ASSERT_TRUE(f.Mid(4, 4).IsOK());
    ASSERT_TRUE(f.Tail(slot).IsOK());
    Check(cudaDeviceSynchronize());
    EXPECT_EQ(f.ctl.Read()[slot], 10);
    EXPECT_EQ(f.ctl.Read()[5], 2);
    EXPECT_EQ(f.ring.Read()[4], 7);
    EXPECT_EQ(f.ring.Read()[0], 1);
  }
}

TEST_F(FoldContribV2Test, CapturedGraphReplaysWithFreshAcceptance) {
  Fixture f;
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  Check(cudaStreamCreate(&stream));
  f.Reset(4, 0);
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  ASSERT_TRUE(f.Mid(4, 4, stream).IsOK());
  ASSERT_TRUE(f.Tail(2, stream).IsOK());
  Check(cudaStreamEndCapture(stream, &graph));
  Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int accepted : {0, 3, 1, 2}) {
    f.Reset(4, accepted);
    Check(cudaGraphLaunch(executable, stream));
    Check(cudaStreamSynchronize(stream));
    EXPECT_EQ(f.selection.Read()[0], accepted);
    EXPECT_EQ(f.ring.Read()[0], 1);
  }
  Check(cudaGraphExecDestroy(executable));
  Check(cudaGraphDestroy(graph));
  Check(cudaStreamDestroy(stream));
}
}  // namespace
}  // namespace onnxruntime::test
