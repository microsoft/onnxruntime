// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <stdexcept>
#include <vector>
#include "contrib_ops/cuda/llm/speculative_fold.h"

namespace onnxruntime::test {
namespace {
namespace fold = contrib::cuda::speculative_fold;

constexpr int kMaxRows = 8;
constexpr int kCtlWords = 16;
constexpr int kRecords = 2;
constexpr int64_t kUntouched = -7;

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
    ASSERT_LE(values.size(), count);
    Check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> Read() const {
    std::vector<T> values(count);
    Check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
  T* data{};
  size_t count;
};

int64_t Prediction(int b, int k) { return 1000 * b + 10 * (k + 1); }
int64_t PrefixWord(int b, int k, int word) { return 1000 * b + 100 * (k + 1) + word; }
int32_t Low(int64_t word) { return static_cast<int32_t>(static_cast<uint64_t>(word) & 0xffffffffu); }
int32_t High(int64_t word) { return static_cast<int32_t>(static_cast<uint64_t>(word) >> 32); }

// Batch item b has length 3 + b, cache length 2 + b and ring sequence `sequence + b`. All ring records start as -7.
struct Fixture {
  explicit Fixture(int batch = 1)
      : batch(batch),
        tokens(batch * kMaxRows),
        ids(batch * (kMaxRows + 4)),
        ctl(batch * kCtlWords),
        ctl_out(batch * kCtlWords),
        context(batch * 2),
        context_out(batch * 2),
        prefixes(batch * 2 * kMaxRows),
        ptrs(2),
        ring(batch * kRecords * 8),
        status(2),
        head_c(batch) {}

  void Reset(int rows, const std::vector<int>& accepted, const std::vector<bool>& muted, int64_t sequence = 0,
             int ids_padding = 0) {
    ids_stride = rows + ids_padding;
    std::vector<int64_t> predictions, input, control(batch * kCtlWords), contexts, prefix_values;
    for (int b = 0; b < batch; ++b) {
      for (int k = 0; k < rows; ++k) predictions.push_back(Prediction(b, k));
      std::vector<int64_t> item(ids_stride, -1);
      item[0] = 99;
      for (int i = 0; i < accepted[b]; ++i) item[i + 1] = Prediction(b, i);
      input.insert(input.end(), item.begin(), item.end());
      for (int w = 0; w < kCtlWords; ++w) control[b * kCtlWords + w] = 5000 + w;
      control[b * kCtlWords + 1] = 3 + b;
      control[b * kCtlWords + 3] = muted[b] ? 1 : 0;
      control[b * kCtlWords + 4] = sequence + b;
      control[b * kCtlWords + 5] = 0;
      control[b * kCtlWords + 6] = 0;
      control[b * kCtlWords + 7] = 2 + b;
      contexts.insert(contexts.end(), {-1 - b, -2 - b});
      for (int k = 0; k < kMaxRows; ++k) prefix_values.insert(prefix_values.end(), {PrefixWord(b, k, 0), PrefixWord(b, k, 1)});
    }
    tokens.Upload(predictions);
    ids.Upload(input);
    ctl.Upload(control);
    ctl_out.Upload(std::vector<int64_t>(batch * kCtlWords, -99));
    context.Upload(contexts);
    context_out.Upload(std::vector<int64_t>(batch * 2, -99));
    prefixes.Upload(prefix_values);
    ptrs.Upload({reinterpret_cast<int64_t>(ring.data), kRecords});
    ring.Upload(std::vector<int64_t>(batch * kRecords * 8, kUntouched));
    status.Upload({0, 0});
  }

  void Reset(int rows, int accepted, bool muted = false, int64_t sequence = 0) {
    Reset(rows, std::vector<int>(batch, accepted), std::vector<bool>(batch, muted), sequence);
  }

  Status Run(int rows, int head_rows, bool in_place = true, cudaStream_t stream = nullptr) {
    in_place_ = in_place;
    return fold::LaunchPrefixCommit(tokens.data, ids.data, status.data, 2, ctl.data,
                                    in_place ? ctl.data : ctl_out.data, context.data,
                                    in_place ? context.data : context_out.data, prefixes.data, ptrs.data,
                                    head_c.data, rows, head_rows, batch, ids_stride, kCtlWords, 2 * kMaxRows, stream);
  }

  std::vector<int64_t> Ctl() const { return in_place_ ? ctl.Read() : ctl_out.Read(); }
  std::vector<int64_t> Context() const { return in_place_ ? context.Read() : context_out.Read(); }

  int batch;
  int ids_stride{kMaxRows};
  bool in_place_{true};
  Buffer<int64_t> tokens, ids, ctl, ctl_out, context, context_out, prefixes, ptrs, ring;
  Buffer<int32_t> status, head_c;
};

class FoldContribTest : public testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "CUDA is required";
  }
};

void ExpectCommitted(const Fixture& f, int rows, int b, int accepted, int64_t sequence) {
  const auto ctl = f.Ctl();
  const int64_t* c = ctl.data() + b * kCtlWords;
  EXPECT_EQ(c[0], Prediction(b, accepted));
  EXPECT_EQ(c[1], 3 + b + accepted + 1);
  EXPECT_EQ(c[3], 0);
  EXPECT_EQ(c[4], sequence + b + 1);
  EXPECT_EQ(c[5], 0);
  EXPECT_EQ(c[6], accepted);
  EXPECT_EQ(c[7], 2 + b + accepted + 1);
  for (int w : {2, 8, 9, 10, 11, 12, 13, 14, 15}) EXPECT_EQ(c[w], 5000 + w) << "word " << w;
  const auto context = f.Context();
  EXPECT_EQ(context[2 * b], PrefixWord(b, accepted, 0));
  EXPECT_EQ(context[2 * b + 1], PrefixWord(b, accepted, 1));
  EXPECT_EQ(f.head_c.Read()[b], 2 + b);

  const auto ring = f.ring.Read();
  const int64_t* record = ring.data() + (b * kRecords + (sequence + b) % kRecords) * 8;
  EXPECT_EQ(record[0], sequence + b + 1);
  EXPECT_EQ(Low(record[1]), Prediction(b, 0));
  EXPECT_EQ(Low(record[2]), Prediction(b, 1));
  EXPECT_EQ(Low(record[3]), accepted);
  EXPECT_EQ(record[4], 0);
  EXPECT_EQ(Low(record[5]), Prediction(b, accepted));
  EXPECT_EQ(record[6], 3 + b + accepted + 1);
  EXPECT_EQ(Low(record[7]), Prediction(b, 2));
  EXPECT_EQ(High(record[7]), rows > 3 ? Prediction(b, 3) : 0);
  EXPECT_EQ(High(record[1]), rows > 4 ? Prediction(b, 4) : 0);
  EXPECT_EQ(High(record[2]), rows > 5 ? Prediction(b, 5) : 0);
  EXPECT_EQ(High(record[3]), rows > 6 ? Prediction(b, 6) : 0);
  EXPECT_EQ(High(record[5]), rows > 7 ? Prediction(b, 7) : 0);
}

TEST_F(FoldContribTest, AllPrefixLengthsAndDirectRingPublication) {
  Fixture f;
  for (bool in_place : {true, false}) {
    for (int rows : {3, 4, 8}) {
      for (int accepted = 0; accepted < rows; ++accepted) {
        SCOPED_TRACE(testing::Message() << "in_place=" << in_place << " rows=" << rows << " accepted=" << accepted);
        f.Reset(rows, accepted, false, 3);
        ASSERT_TRUE(f.Run(rows, rows, in_place).IsOK());
        Check(cudaDeviceSynchronize());
        ExpectCommitted(f, rows, 0, accepted, 3);
      }
    }
  }
}

TEST_F(FoldContribTest, SeparateOutputsLeaveInputStateUntouched) {
  Fixture f;
  f.Reset(8, 5);
  const auto ctl_before = f.ctl.Read();
  const auto context_before = f.context.Read();
  ASSERT_TRUE(f.Run(8, 8, false).IsOK());
  Check(cudaDeviceSynchronize());
  EXPECT_EQ(f.ctl.Read(), ctl_before);
  EXPECT_EQ(f.context.Read(), context_before);
  ExpectCommitted(f, 8, 0, 5, 0);
}

TEST_F(FoldContribTest, BatchedRequestsUsePerItemStateAndRing) {
  constexpr int kBatch = 3;
  const std::vector<int> accepted{3, 0, 1};
  const std::vector<bool> muted{false, true, false};
  Fixture f(kBatch);
  for (bool in_place : {true, false}) {
    f.Reset(kMaxRows, accepted, muted, 1, /*ids_padding=*/3);
    ASSERT_TRUE(f.Run(kMaxRows, kMaxRows, in_place).IsOK());
    Check(cudaDeviceSynchronize());
    ExpectCommitted(f, kMaxRows, 0, accepted[0], 1);
    ExpectCommitted(f, kMaxRows, 2, accepted[2], 1);
    const auto ctl = f.Ctl();
    EXPECT_EQ(ctl[kCtlWords + 4], 2);
    EXPECT_EQ(ctl[kCtlWords + 0], 5000);
    EXPECT_EQ(f.Context()[2], -2);
    EXPECT_EQ(f.Context()[3], -3);
    EXPECT_EQ(f.head_c.Read()[1], 0);
    const auto ring = f.ring.Read();
    for (int r = 0; r < kRecords * 8; ++r) EXPECT_EQ(ring[kRecords * 8 + r], kUntouched);
  }
}

TEST_F(FoldContribTest, MutedPrefillDoesNotAccessRingOrCommitContext) {
  Fixture f;
  for (bool in_place : {true, false}) {
    f.Reset(8, 0, true);
    f.ptrs.Upload({0, 0});
    // Muted prefill may use any number of rows; the kernel never reads the tokens.
    ASSERT_TRUE(f.Run(64, 8, in_place).IsOK());
    Check(cudaDeviceSynchronize());
    EXPECT_EQ(f.Context(), (std::vector<int64_t>{-1, -2}));
    const auto ctl = f.Ctl();
    EXPECT_EQ(ctl[1], 3);
    EXPECT_EQ(ctl[3], 1);
    EXPECT_EQ(ctl[4], 0);
    EXPECT_EQ(ctl[5], 0);
    EXPECT_EQ(ctl[7], 2);
    EXPECT_EQ(f.head_c.Read()[0], 0);
    EXPECT_EQ(f.ring.Read(), std::vector<int64_t>(kRecords * 8, kUntouched));
  }
}

TEST_F(FoldContribTest, InvalidDeviceMetadataMutesBeforeRingAccess) {
  Fixture f;
  for (int kind = 0; kind < 6; ++kind) {
    SCOPED_TRACE(kind);
    f.Reset(4, 3);
    auto ctl = f.ctl.Read();
    int64_t bit = 16;
    int head_rows = 4;
    if (kind == 0) f.ptrs.Upload({0, 0});
    if (kind == 1) ctl[4] = INT64_MAX;
    if (kind == 2) f.ptrs.Upload({reinterpret_cast<int64_t>(f.ring.data), INT64_MAX});
    if (kind >= 3) bit = 4;
    if (kind == 3) ctl[7] = -1;
    if (kind == 4) ctl[1] = -1;
    if (kind == 5) head_rows = 5;
    f.ctl.Upload(ctl);
    ASSERT_TRUE(f.Run(4, head_rows).IsOK());
    Check(cudaDeviceSynchronize());
    EXPECT_EQ(f.head_c.Read()[0], 0);
    EXPECT_EQ(f.Ctl()[5] & bit, bit);
    EXPECT_EQ(f.Ctl()[3], 1);
    EXPECT_EQ(f.Context(), (std::vector<int64_t>{-1, -2}));
    EXPECT_EQ(f.ring.Read(), std::vector<int64_t>(kRecords * 8, kUntouched));
  }
}

TEST_F(FoldContribTest, ValidationStatusIsPublished) {
  Fixture f;
  for (const auto& statuses : {std::vector<int32_t>{0, 0}, std::vector<int32_t>{1, 4}}) {
    f.Reset(4, 2);
    f.status.Upload(statuses);
    ASSERT_TRUE(f.Run(4, 4).IsOK());
    Check(cudaDeviceSynchronize());
    const int32_t combined = statuses[0] | statuses[1];
    EXPECT_EQ(f.Ctl()[5], combined ? 2 : 0);
    EXPECT_EQ(f.ring.Read()[4], combined);
    EXPECT_EQ(f.ring.Read()[0], 1);
  }
}

TEST_F(FoldContribTest, CapturedGraphReplaysWithFreshAcceptance) {
  Fixture f;
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  Check(cudaStreamCreate(&stream));
  f.Reset(4, 0);
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  ASSERT_TRUE(f.Run(4, 4, true, stream).IsOK());
  Check(cudaStreamEndCapture(stream, &graph));
  Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (int accepted : {0, 3, 1, 2}) {
    f.Reset(4, accepted);
    Check(cudaGraphLaunch(executable, stream));
    Check(cudaStreamSynchronize(stream));
    ExpectCommitted(f, 4, 0, accepted, 0);
  }
  Check(cudaGraphExecDestroy(executable));
  Check(cudaGraphDestroy(graph));
  Check(cudaStreamDestroy(stream));
}
}  // namespace
}  // namespace onnxruntime::test
