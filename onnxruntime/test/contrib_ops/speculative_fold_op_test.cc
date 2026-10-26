// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && !defined(ORT_MINIMAL_BUILD)

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {
namespace {

class FoldResultRing {
 public:
  FoldResultRing() {
    const cudaError_t error = cudaHostAlloc(reinterpret_cast<void**>(&host_), 32 * sizeof(int64_t), cudaHostAllocMapped);
    ORT_ENFORCE(error == cudaSuccess, cudaGetErrorString(error));
    const cudaError_t pointer_error = cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_), host_, 0);
    if (pointer_error != cudaSuccess) {
      cudaFreeHost(host_);
      ORT_THROW(cudaGetErrorString(pointer_error));
    }
    std::fill_n(host_, 32, int64_t{-1});
  }

  ~FoldResultRing() { cudaFreeHost(host_); }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FoldResultRing);

  int64_t Address() const { return static_cast<int64_t>(reinterpret_cast<intptr_t>(device_)); }
  int64_t* Host() const { return host_; }

 private:
  int64_t* host_ = nullptr;
  int64_t* device_ = nullptr;
};

TEST(SpeculativeFoldTest, InferenceSessionLoadsVersionOneAndAcceptsMutedPrefillRows) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80 or newer";
  auto provider = DefaultCudaExecutionProvider();
  ASSERT_NE(provider, nullptr);
  FoldResultRing ring;

  Model model("speculative_fold", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto arg = [&](const char* name, int element_type, std::initializer_list<int64_t> dims) -> NodeArg* {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor = type.mutable_tensor_type();
    tensor->set_elem_type(element_type);
    for (const auto dim : dims) {
      auto* shape_dim = tensor->mutable_shape()->add_dim();
      if (dim < 0) {
        shape_dim->set_dim_param("rows");
      } else {
        shape_dim->set_dim_value(dim);
      }
    }
    return &graph.GetOrCreateNodeArg(name, &type);
  };

  constexpr int i64 = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  constexpr int i32 = ONNX_NAMESPACE::TensorProto_DataType_INT32;
  constexpr int bf16 = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  auto* tokens = arg("tokens", i64, {-1});
  auto* ids = arg("ids", i64, {-1});
  auto* status = arg("status", i32, {1});
  auto* ctl = arg("ctl", i64, {8});
  auto* context = arg("context", i64, {2});
  auto* prefixes = arg("prefixes", i64, {8});
  auto* prefix2 = arg("prefix2", i64, {2});
  auto* ptrs = arg("ptrs", i64, {2});
  auto* cosine = arg("cosine", bf16, {16, 3});
  auto* sine = arg("sine", bf16, {16, 3});
  auto* head_ids = arg("head_ids", i64, {1, 2});
  auto* head_c = arg("head_c", i32, {1});
  auto* slots = arg("slots", i64, {1, 2});
  auto* lengths = arg("lengths", i32, {1});
  auto* cos_out = arg("cos_out", bf16, {1, 2, 3});
  auto* sin_out = arg("sin_out", bf16, {1, 2, 3});
  auto* selected = arg("selected", i64, {1});
  auto* updated_ctl = arg("updated_ctl", i64, {8});
  auto* updated_context = arg("updated_context", i64, {2});
  graph.AddNode("mid", "SpeculativePrefixCommit", "",
                {tokens, ids, status, ctl, context, prefixes, prefix2, ptrs, cosine, sine},
                {head_ids, head_c, slots, lengths, cos_out, sin_out, selected, updated_ctl, updated_context},
                nullptr, kMSDomain)
      .SetExecutionProviderType(kCudaExecutionProvider);
  graph.SetOutputs({head_ids, head_c, slots, lengths, cos_out, sin_out, selected,
                    updated_ctl, updated_context});
  ASSERT_STATUS_OK(graph.Resolve());

  std::stringstream stream;
  ASSERT_TRUE(model.ToProto().SerializeToOstream(&stream));
  SessionOptions options;
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  ASSERT_STATUS_OK(session.Load(stream));
  ASSERT_STATUS_OK(session.Initialize());

  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  NameMLValMap feeds;
  auto feed = [&](const char* name, const TensorShapeVector& shape, const auto& values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    OrtValue value;
    CreateMLValue<T>(allocator, shape, values, &value);
    feeds[name] = std::move(value);
  };
  std::vector<BFloat16> rope(16 * 3);
  for (size_t i = 0; i < rope.size(); ++i) rope[i] = BFloat16(static_cast<float>(i));

  auto run = [&](int rows, bool muted) {
    std::vector<int64_t> token_values(rows);
    std::vector<int64_t> id_values(rows, -1);
    for (int row = 0; row < rows; ++row) token_values[row] = row + 10;
    id_values[0] = 0;
    if (rows > 1) id_values[1] = token_values[0];
    feed("tokens", {rows}, token_values);
    feed("ids", {rows}, id_values);
    feed("status", {1}, std::vector<int32_t>{0});
    feed("ctl", {8}, std::vector<int64_t>{0, 3, 0, muted ? 1 : 0, 0, 0, 0, 2});
    feed("context", {2}, std::vector<int64_t>{1, 2});
    feed("prefixes", {8}, std::vector<int64_t>{100, 101, 200, 201, 300, 301, 400, 401});
    feed("prefix2", {2}, std::vector<int64_t>{200, 201});
    feed("ptrs", {2}, muted ? std::vector<int64_t>{0, 0} : std::vector<int64_t>{ring.Address(), 4});
    feed("cosine", {16, 3}, rope);
    feed("sine", {16, 3}, rope);
    std::vector<OrtValue> outputs;
    const std::vector<std::string> output_names{
        "head_ids", "head_c", "slots", "lengths", "cos_out", "sin_out",
        "selected", "updated_ctl", "updated_context"};
    const Status status = session.Run(RunOptions{}, feeds, output_names, &outputs);
    EXPECT_TRUE(status.IsOK()) << status;
    if (status.IsOK()) {
      EXPECT_EQ(outputs[6].Get<Tensor>().Data<int64_t>()[0], muted ? 0 : 1);
      EXPECT_EQ(outputs[1].Get<Tensor>().Data<int32_t>()[0], muted ? 0 : 2);
    }
  };

  run(2, false);
  run(3, true);
  run(4, true);
}

TEST(SpeculativeFoldTest, SupportsBatchedSpeculativePrefixCommitS8) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80 or newer";
  auto provider = DefaultCudaExecutionProvider();
  ASSERT_NE(provider, nullptr);

  constexpr int kBatch = 3;
  constexpr int kRows = 8;
  constexpr int kRingCap = 4;
  int64_t* ring_host = nullptr;
  int64_t* ring_dev = nullptr;
  ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void**>(&ring_host), kBatch * kRingCap * 8 * sizeof(int64_t), cudaHostAllocMapped),
            cudaSuccess);
  ASSERT_EQ(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ring_dev), ring_host, 0), cudaSuccess);
  std::fill_n(ring_host, kBatch * kRingCap * 8, int64_t{-1});

  Model model("speculative_fold_batch", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto arg = [&](const char* name, int element_type, std::initializer_list<int64_t> dims) -> NodeArg* {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor = type.mutable_tensor_type();
    tensor->set_elem_type(element_type);
    int axis = 0;
    for (const auto dim : dims) {
      auto* shape_dim = tensor->mutable_shape()->add_dim();
      if (dim < 0) {
        shape_dim->set_dim_param(axis == 0 ? "batch" : "rows");
      } else {
        shape_dim->set_dim_value(dim);
      }
      ++axis;
    }
    return &graph.GetOrCreateNodeArg(name, &type);
  };

  constexpr int i64 = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  constexpr int i32 = ONNX_NAMESPACE::TensorProto_DataType_INT32;
  constexpr int bf16 = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  auto* tokens = arg("tokens", i64, {-1, kRows});
  auto* ids = arg("ids", i64, {-1, kRows});
  auto* status = arg("status", i32, {1});
  auto* ctl = arg("ctl", i64, {kBatch * 16});
  auto* context = arg("context", i64, {kBatch * 2});
  auto* prefixes = arg("prefixes", i64, {kBatch * 16});
  auto* prefix2 = arg("prefix2", i64, {2});
  auto* ptrs = arg("ptrs", i64, {2});
  auto* cosine = arg("cosine", bf16, {64, 4});
  auto* sine = arg("sine", bf16, {64, 4});
  auto* head_ids = arg("head_ids", i64, {-1, kRows});
  auto* head_c = arg("head_c", i32, {-1});
  auto* slots = arg("slots", i64, {-1, kRows});
  auto* lengths = arg("lengths", i32, {-1});
  auto* cos_out = arg("cos_out", bf16, {-1, kRows, 4});
  auto* sin_out = arg("sin_out", bf16, {-1, kRows, 4});
  auto* selected = arg("selected", i64, {-1});
  auto* updated_ctl = arg("updated_ctl", i64, {kBatch * 16});
  auto* updated_context = arg("updated_context", i64, {kBatch * 2});

  ONNX_NAMESPACE::AttributeProto head_rows_attr;
  head_rows_attr.set_name("head_rows");
  head_rows_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  head_rows_attr.set_i(kRows);
  NodeAttributes attrs{{"head_rows", head_rows_attr}};

  graph.AddNode("mid", "SpeculativePrefixCommit", "",
                {tokens, ids, status, ctl, context, prefixes, prefix2, ptrs, cosine, sine},
                {head_ids, head_c, slots, lengths, cos_out, sin_out, selected, updated_ctl, updated_context},
                &attrs, kMSDomain)
      .SetExecutionProviderType(kCudaExecutionProvider);
  graph.SetOutputs({head_ids, head_c, slots, lengths, cos_out, sin_out, selected,
                    updated_ctl, updated_context});
  ASSERT_STATUS_OK(graph.Resolve());

  std::stringstream stream;
  ASSERT_TRUE(model.ToProto().SerializeToOstream(&stream));
  SessionOptions options;
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  ASSERT_STATUS_OK(session.Load(stream));
  ASSERT_STATUS_OK(session.Initialize());

  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  NameMLValMap feeds;
  auto feed = [&](const char* name, const TensorShapeVector& shape, const auto& values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    OrtValue value;
    CreateMLValue<T>(allocator, shape, values, &value);
    feeds[name] = std::move(value);
  };
  std::vector<BFloat16> rope(64 * 4);
  for (size_t i = 0; i < rope.size(); ++i) rope[i] = BFloat16(static_cast<float>(i));

  // Batch 0: accepts 3 drafts (tokens[0..2] == ids[1..3], tokens[3] != ids[4])
  // Batch 1: muted (ctl[3] = 1)
  // Batch 2: accepts 1 draft (tokens[0] == ids[1], tokens[1] != ids[2])
  std::vector<int64_t> token_values = {
      10,
      11,
      12,
      99,
      14,
      15,
      16,
      17,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      30,
      88,
      32,
      33,
      34,
      35,
      36,
      37,
  };
  std::vector<int64_t> id_values = {
      1,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      2,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      3,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
  };
  std::vector<int64_t> ctl_values(kBatch * 16, 0);
  // Batch 0: length=10, muted=0, seq=0, cache_len=10
  ctl_values[0 * 16 + 1] = 10;
  ctl_values[0 * 16 + 7] = 10;
  // Batch 1: length=20, muted=1, seq=5, cache_len=20
  ctl_values[1 * 16 + 1] = 20;
  ctl_values[1 * 16 + 3] = 1;
  ctl_values[1 * 16 + 4] = 5;
  ctl_values[1 * 16 + 7] = 20;
  // Batch 2: length=30, muted=0, seq=2, cache_len=30
  ctl_values[2 * 16 + 1] = 30;
  ctl_values[2 * 16 + 4] = 2;
  ctl_values[2 * 16 + 7] = 30;

  std::vector<int64_t> prefixes_values(kBatch * 16);
  for (int b = 0; b < kBatch; ++b) {
    for (int r = 0; r < kRows; ++r) {
      prefixes_values[b * 16 + 2 * r + 0] = 1000 * (b + 1) + 10 * r + 0;
      prefixes_values[b * 16 + 2 * r + 1] = 1000 * (b + 1) + 10 * r + 1;
    }
  }

  feed("tokens", {kBatch, kRows}, token_values);
  feed("ids", {kBatch, kRows}, id_values);
  feed("status", {1}, std::vector<int32_t>{0});
  feed("ctl", {kBatch * 16}, ctl_values);
  feed("context", {kBatch * 2}, std::vector<int64_t>{1, 2, 3, 4, 5, 6});
  feed("prefixes", {kBatch * 16}, prefixes_values);
  feed("prefix2", {2}, std::vector<int64_t>{prefixes_values[2], prefixes_values[3]});
  feed("ptrs", {2}, std::vector<int64_t>{reinterpret_cast<int64_t>(ring_dev), kRingCap});
  feed("cosine", {64, 4}, rope);
  feed("sine", {64, 4}, rope);

  std::vector<OrtValue> outputs;
  const std::vector<std::string> output_names{
      "head_ids", "head_c", "slots", "lengths", "cos_out", "sin_out",
      "selected", "updated_ctl", "updated_context"};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));

  const int64_t* sel = outputs[6].Get<Tensor>().Data<int64_t>();
  EXPECT_EQ(sel[0], 3);
  EXPECT_EQ(sel[1], 0);  // muted
  EXPECT_EQ(sel[2], 1);

  const int64_t* out_ctl = outputs[7].Get<Tensor>().Data<int64_t>();
  EXPECT_EQ(out_ctl[0 * 16 + 0], 99);      // pending = tokens[0, 3]
  EXPECT_EQ(out_ctl[0 * 16 + 1], 10 + 4);  // length = 10 + 3 + 1
  EXPECT_EQ(out_ctl[0 * 16 + 4], 1);       // seq incremented to 1

  EXPECT_EQ(out_ctl[1 * 16 + 4], 5);  // muted: seq unchanged

  EXPECT_EQ(out_ctl[2 * 16 + 0], 88);      // pending = tokens[2, 1]
  EXPECT_EQ(out_ctl[2 * 16 + 1], 30 + 2);  // length = 30 + 1 + 1
  EXPECT_EQ(out_ctl[2 * 16 + 4], 3);       // seq incremented from 2 to 3

  const int64_t* out_ctx = outputs[8].Get<Tensor>().Data<int64_t>();
  // Batch 0 accepted=3 -> prefixes[0, 6..7]
  EXPECT_EQ(out_ctx[0], 1030);
  EXPECT_EQ(out_ctx[1], 1031);
  // Batch 2 accepted=1 -> prefixes[2, 2..3]
  EXPECT_EQ(out_ctx[4], 3010);
  EXPECT_EQ(out_ctx[5], 3011);

  // Ring records published for batch 0 (slot 0) and batch 2 (slot 2)
  EXPECT_EQ(ring_host[(0 * kRingCap + 0) * 8 + 0], 1);
  EXPECT_EQ(ring_host[(1 * kRingCap + 1) * 8 + 0], -1);  // muted: untouched
  EXPECT_EQ(ring_host[(2 * kRingCap + 2) * 8 + 0], 3);

  cudaFreeHost(ring_host);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
