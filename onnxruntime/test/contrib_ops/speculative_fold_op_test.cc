// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && !defined(ORT_MINIMAL_BUILD)

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "core/common/logging/logging.h"
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
    cudaError_t error = cudaHostAlloc(reinterpret_cast<void**>(&host_), 32 * sizeof(int64_t), cudaHostAllocMapped);
    ORT_ENFORCE(error == cudaSuccess, cudaGetErrorString(error));
    error = cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_), host_, 0);
    if (error != cudaSuccess) {
      cudaFreeHost(host_);
      ORT_THROW(cudaGetErrorString(error));
    }
    std::fill_n(host_, 32, int64_t{-1});
  }
  ~FoldResultRing() { cudaFreeHost(host_); }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FoldResultRing);
  int64_t* Host() const { return host_; }
  int64_t Address() const { return static_cast<int64_t>(reinterpret_cast<intptr_t>(device_)); }

 private:
  int64_t* host_ = nullptr;
  int64_t* device_ = nullptr;
};

TEST(SpeculativeFoldTest, CpuFeedsPropagateStateAcrossRunsAndRejectInvalidMetadata) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80 or newer";
  auto provider = DefaultCudaExecutionProvider();
  ASSERT_NE(provider, nullptr);
  FoldResultRing ring;
  Model model("fold_state_outputs", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto arg = [&](const char* name, int element_type, std::initializer_list<int64_t> shape) -> NodeArg* {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor = type.mutable_tensor_type();
    tensor->set_elem_type(element_type);
    for (auto dim : shape) tensor->mutable_shape()->add_dim()->set_dim_value(dim);
    return &graph.GetOrCreateNodeArg(name, &type);
  };
  constexpr int i64 = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  constexpr int i32 = ONNX_NAMESPACE::TensorProto_DataType_INT32;
  constexpr int bf16 = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  auto* next_tokens = arg("next_tokens", i64, {2});
  auto* input_ids = arg("input_ids", i64, {2});
  auto* status = arg("status", i32, {1});
  auto* ctl = arg("ctl", i64, {8});
  auto* ctx = arg("ctx_commit", i64, {2});
  auto* ctx_a = arg("ctx_a", i64, {2});
  auto* ctx_b = arg("ctx_b", i64, {2});
  auto* ptrs = arg("ptrs", i64, {2});
  auto* cos = arg("cos_cache", bf16, {16, 32});
  auto* sin = arg("sin_cache", bf16, {16, 32});
  auto* head_ids = arg("head_ids", i64, {1, 2});
  auto* head_c = arg("head_c", i32, {1});
  auto* slots = arg("slot_ids", i64, {1, 2});
  auto* lengths = arg("seqlens", i32, {1});
  auto* cos_out = arg("cos_out", bf16, {1, 2, 32});
  auto* sin_out = arg("sin_out", bf16, {1, 2, 32});
  auto* selected = arg("sel", i64, {1});
  auto* mid_ctl = arg("mid_ctl", i64, {8});
  auto* updated_ctx = arg("updated_ctx", i64, {2});
  auto* draft = arg("draft", i64, {1});
  auto* done = arg("done", i32, {1});
  auto* updated_ctl = arg("updated_ctl", i64, {8});
  graph.AddNode("mid", "FoldMid2", "",
                {next_tokens, input_ids, status, ctl, ctx, ctx_a, ctx_b, ptrs, cos, sin},
                {head_ids, head_c, slots, lengths, cos_out, sin_out, selected, mid_ctl, updated_ctx},
                nullptr, kMSDomain)
      .SetExecutionProviderType(kCudaExecutionProvider);
  graph.AddNode("tail", "FoldTail", "", {draft, status, mid_ctl}, {done, updated_ctl}, nullptr, kMSDomain)
      .SetExecutionProviderType(kCudaExecutionProvider);
  graph.SetOutputs({updated_ctl, updated_ctx, selected, cos_out, sin_out, done});
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
  const std::vector<std::string> output_names{
      "updated_ctl", "updated_ctx", "sel", "cos_out", "sin_out", "done"};
  auto feed = [&](const char* name, const TensorShapeVector& shape, const auto& values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    OrtValue value;
    CreateMLValue<T>(allocator, shape, values, &value);
    feeds[name] = std::move(value);
  };
  feed("next_tokens", {2}, std::vector<int64_t>{7, 8});
  feed("input_ids", {2}, std::vector<int64_t>{0, 7});
  feed("status", {1}, std::vector<int32_t>{4});
  feed("ctx_a", {2}, std::vector<int64_t>{11, 12});
  feed("ctx_b", {2}, std::vector<int64_t>{21, 22});
  feed("ptrs", {2}, std::vector<int64_t>{ring.Address(), 4});
  feed("draft", {1}, std::vector<int64_t>{42});
  std::vector<BFloat16> cache(16 * 32);
  for (size_t i = 0; i < cache.size(); ++i) cache[i] = BFloat16(static_cast<float>(i));
  feed("cos_cache", {16, 32}, cache);
  feed("sin_cache", {16, 32}, cache);
  std::vector<int64_t> control{0, 10, 99, 0, 0, 0, 0, 5};
  std::vector<int64_t> context{1, 2};
  for (int run = 0; run < 2; ++run) {
    feed("ctl", {8}, control);
    feed("ctx_commit", {2}, context);
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));
    ASSERT_EQ(outputs.size(), 6u);
    const auto* result = outputs[0].Get<Tensor>().Data<int64_t>();
    EXPECT_EQ(result[0], 8);
    EXPECT_EQ(result[1], control[1] + 2);
    EXPECT_EQ(result[2], 42);
    EXPECT_EQ(result[4], control[4] + 1);
    EXPECT_EQ(result[5], 2);
    EXPECT_EQ(result[7], control[7] + 2);
    EXPECT_EQ(outputs[2].Get<Tensor>().Data<int64_t>()[0], 1);
    EXPECT_EQ(outputs[5].Get<Tensor>().Data<int32_t>()[0], 1);
    for (int output_index : {3, 4}) {
      const auto* rotary = outputs[output_index].Get<Tensor>().Data<BFloat16>();
      for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(rotary[i], cache[static_cast<size_t>(control[1] + 1) * 32 + i]);
      }
    }
    const auto* committed = outputs[1].Get<Tensor>().Data<int64_t>();
    EXPECT_EQ(committed[0], 21);
    EXPECT_EQ(committed[1], 22);
    EXPECT_TRUE(std::equal(control.begin(), control.end(), feeds.at("ctl").Get<Tensor>().Data<int64_t>()));
    EXPECT_TRUE(std::equal(context.begin(), context.end(), feeds.at("ctx_commit").Get<Tensor>().Data<int64_t>()));
    control.assign(result, result + 8);
    context.assign(committed, committed + 2);
  }
  EXPECT_EQ(control[1], 14);
  EXPECT_EQ(ring.Host()[0], 1);
  EXPECT_EQ(ring.Host()[8], 2);

  // Rejection at the last valid pair still commits one token; a muted run preserves both states.
  feed("input_ids", {2}, std::vector<int64_t>{0, 9});
  for (int64_t muted : {int64_t{0}, int64_t{1}}) {
    control = {0, 13, 99, muted, 2, 0, 0, 5};
    context = {1, 2};
    feed("ctl", {8}, control);
    feed("ctx_commit", {2}, context);
    feed("ptrs", {2}, std::vector<int64_t>{muted ? 0 : ring.Address(), muted ? 0 : 4});
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));
    const auto* result = outputs[0].Get<Tensor>().Data<int64_t>();
    EXPECT_EQ(outputs[2].Get<Tensor>().Data<int64_t>()[0], 0);
    EXPECT_EQ(outputs[5].Get<Tensor>().Data<int32_t>()[0], muted ? 0 : 1);
    EXPECT_EQ(result[1], muted ? 13 : 14);
    EXPECT_EQ(result[2], muted ? 99 : 42);
    EXPECT_EQ(result[4], muted ? 2 : 3);
    EXPECT_EQ(result[5], muted ? 0 : 2);
    EXPECT_EQ(result[7], muted ? 5 : 6);
    const auto* committed = outputs[1].Get<Tensor>().Data<int64_t>();
    EXPECT_EQ(committed[0], muted ? 1 : 11);
    EXPECT_EQ(committed[1], muted ? 2 : 12);
    for (int output_index : {3, 4}) {
      const auto* rotary = outputs[output_index].Get<Tensor>().Data<BFloat16>();
      for (int i = 0; i < 64; ++i) EXPECT_EQ(rotary[i], cache[14 * 32 + i]);
    }
  }

  struct InvalidCase {
    int64_t position;
    int64_t muted;
    int64_t address;
    int64_t capacity;
    int64_t counter;
    int64_t error;
    int64_t cache_length = 5;
  };
  const std::vector<InvalidCase> invalid_cases{
      {14, 0, ring.Address(), 4, 0, 8}, {14, 1, ring.Address(), 4, 0, 8}, {-1, 0, ring.Address(), 4, 0, 8}, {std::numeric_limits<int64_t>::max(), 0, ring.Address(), 4, 0, 8}, {10, 0, 0, 4, 0, 16}, {10, 0, ring.Address(), 0, 0, 16}, {10, 0, ring.Address(), -1, 0, 16}, {10, 0, ring.Address(), 4, -1, 16}, {10, 0, ring.Address(), 4, std::numeric_limits<int64_t>::max(), 16}, {10, 0, ring.Address() + 1, 4, 0, 16}, {10, 0, ring.Address(), std::numeric_limits<int64_t>::max(), 0, 16}, {10, 0, ring.Address(), 4, 0, 4, -1}, {10, 0, ring.Address(), 4, 0, 4, 33027}};
  const std::vector<int64_t> preserved_ring(ring.Host(), ring.Host() + 32);
  for (const auto& invalid : invalid_cases) {
    SCOPED_TRACE(invalid.position);
    SCOPED_TRACE(invalid.capacity);
    SCOPED_TRACE(invalid.counter);
    control = {0, invalid.position, 99, invalid.muted, invalid.counter, 0, 0, invalid.cache_length};
    feed("ctl", {8}, control);
    feed("ctx_commit", {2}, context);
    feed("ptrs", {2}, std::vector<int64_t>{invalid.address, invalid.capacity});
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));
    auto expected = control;
    expected[3] = 1;
    expected[5] = invalid.error;
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), outputs[0].Get<Tensor>().Data<int64_t>()));
    EXPECT_TRUE(std::equal(context.begin(), context.end(), outputs[1].Get<Tensor>().Data<int64_t>()));
    EXPECT_EQ(outputs[2].Get<Tensor>().Data<int64_t>()[0], -1);
    EXPECT_EQ(outputs[5].Get<Tensor>().Data<int32_t>()[0], 0);
    for (int output_index : {3, 4}) {
      const auto* rotary = outputs[output_index].Get<Tensor>().Data<BFloat16>();
      for (int i = 0; i < 64; ++i) EXPECT_EQ(rotary[i].ToFloat(), 0.0f);
    }
    EXPECT_TRUE(std::equal(preserved_ring.begin(), preserved_ring.end(), ring.Host()));
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
