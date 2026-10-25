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
  auto* hidden = arg("hidden", bf16, {1, -1, 3});
  auto* head_ids = arg("head_ids", i64, {1, 2});
  auto* head_c = arg("head_c", i32, {1});
  auto* slots = arg("slots", i64, {1, 2});
  auto* lengths = arg("lengths", i32, {1});
  auto* cos_out = arg("cos_out", bf16, {1, 2, 3});
  auto* sin_out = arg("sin_out", bf16, {1, 2, 3});
  auto* selected = arg("selected", i64, {1});
  auto* head_hidden = arg("head_hidden", bf16, {1, 2, 3});
  auto* mid_ctl = arg("mid_ctl", i64, {8});
  auto* updated_context = arg("updated_context", i64, {2});
  auto* draft = arg("draft", i64, {1});
  auto* downstream_status = arg("downstream_status", i32, {1});
  auto* done = arg("done", i32, {1});
  auto* updated_ctl = arg("updated_ctl", i64, {8});
  graph.AddNode("mid", "SpeculativePrefixCommit", "",
                {tokens, ids, status, ctl, context, prefixes, prefix2, ptrs, cosine, sine, hidden},
                {head_ids, head_c, slots, lengths, cos_out, sin_out, selected, head_hidden, mid_ctl, updated_context},
                nullptr, kMSDomain)
      .SetExecutionProviderType(kCudaExecutionProvider);
  graph.AddNode("tail", "SpeculativeStepPublish", "", {draft, downstream_status, mid_ctl, ptrs}, {done, updated_ctl}, nullptr,
                kMSDomain)
      .SetExecutionProviderType(kCudaExecutionProvider);
  graph.SetOutputs({head_ids, head_c, slots, lengths, cos_out, sin_out, selected, head_hidden,
                    updated_ctl, updated_context, done});
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
    std::vector<BFloat16> hidden_rows(rows * 3);
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
    feed("hidden", {1, rows, 3}, hidden_rows);
    feed("draft", {1}, std::vector<int64_t>{42});
    feed("downstream_status", {1}, std::vector<int32_t>{0});
    std::vector<OrtValue> outputs;
    const std::vector<std::string> output_names{
        "head_ids", "head_c", "slots", "lengths", "cos_out", "sin_out",
        "selected", "head_hidden", "updated_ctl", "updated_context", "done"};
    const Status status = session.Run(RunOptions{}, feeds, output_names, &outputs);
    EXPECT_TRUE(status.IsOK()) << status;
    if (status.IsOK()) {
      EXPECT_EQ(outputs[6].Get<Tensor>().Data<int64_t>()[0], muted ? 0 : 1);
      EXPECT_EQ(outputs[10].Get<Tensor>().Data<int32_t>()[0], muted ? 0 : 1);
    }
  };

  run(2, false);
  run(3, true);
  run(4, true);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
