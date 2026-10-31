// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && !defined(ORT_MINIMAL_BUILD)

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "gmock/gmock.h"
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

constexpr int kCtlWords = 16;

// Host-mapped result ring that the kernel can write through its device address.
class FoldResultRing {
 public:
  explicit FoldResultRing(size_t words) : words_(words) {
    const cudaError_t error = cudaHostAlloc(reinterpret_cast<void**>(&host_), words * sizeof(int64_t), cudaHostAllocMapped);
    ORT_ENFORCE(error == cudaSuccess, cudaGetErrorString(error));
    const cudaError_t pointer_error = cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_), host_, 0);
    if (pointer_error != cudaSuccess) {
      cudaFreeHost(host_);
      ORT_THROW(cudaGetErrorString(pointer_error));
    }
    std::fill_n(host_, words, int64_t{-1});
  }

  ~FoldResultRing() { cudaFreeHost(host_); }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FoldResultRing);

  int64_t Address() const { return static_cast<int64_t>(reinterpret_cast<intptr_t>(device_)); }
  const int64_t* Host() const { return host_; }
  size_t Words() const { return words_; }

 private:
  size_t words_;
  int64_t* host_ = nullptr;
  int64_t* device_ = nullptr;
};

// A one-node SpeculativePrefixCommit graph. Outputs are left without types so that inference provides them.
// token_dims == nullopt leaves next_tokens without a shape; negative dims are symbolic.
class FoldModel {
 public:
  FoldModel(std::optional<std::vector<int64_t>> token_dims, int batch, int head_rows = 8)
      : model_("speculative_fold", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
               {{kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger()) {
    auto& graph = model_.MainGraph();
    constexpr int i64 = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    constexpr int i32 = ONNX_NAMESPACE::TensorProto_DataType_INT32;
    auto arg = [&](const char* name, int element_type, const std::optional<std::vector<int64_t>>& dims) {
      ONNX_NAMESPACE::TypeProto type;
      auto* tensor = type.mutable_tensor_type();
      tensor->set_elem_type(element_type);
      if (dims) {
        int axis = 0;
        for (const auto dim : *dims) {
          auto* shape_dim = tensor->mutable_shape()->add_dim();
          if (dim < 0) {
            shape_dim->set_dim_param(axis == 0 ? "batch" : "rows");
          } else {
            shape_dim->set_dim_value(dim);
          }
          ++axis;
        }
      }
      return &graph.GetOrCreateNodeArg(name, &type);
    };
    auto* tokens = arg("tokens", i64, token_dims);
    auto* ids = arg("ids", i64, std::vector<int64_t>{-1});
    auto* status = arg("status", i32, std::vector<int64_t>{1});
    auto* ctl = arg("ctl", i64, std::vector<int64_t>{batch * kCtlWords});
    auto* context = arg("context", i64, std::vector<int64_t>{batch * 2});
    auto* prefixes = arg("prefixes", i64, std::vector<int64_t>{batch * kCtlWords});
    auto* ptrs = arg("ptrs", i64, std::vector<int64_t>{2});
    head_c_ = &graph.GetOrCreateNodeArg("head_c", nullptr);
    auto* updated_ctl = &graph.GetOrCreateNodeArg("updated_ctl", nullptr);
    auto* updated_context = &graph.GetOrCreateNodeArg("updated_context", nullptr);

    ONNX_NAMESPACE::AttributeProto head_rows_attr;
    head_rows_attr.set_name("head_rows");
    head_rows_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    head_rows_attr.set_i(head_rows);
    NodeAttributes attrs{{"head_rows", head_rows_attr}};
    graph.AddNode("commit", "SpeculativePrefixCommit", "",
                  {tokens, ids, status, ctl, context, prefixes, ptrs},
                  {head_c_, updated_ctl, updated_context}, &attrs, kMSDomain)
        .SetExecutionProviderType(kCudaExecutionProvider);
    graph.SetOutputs({head_c_, updated_ctl, updated_context});
    resolve_status_ = graph.Resolve();
  }

  const Status& ResolveStatus() const { return resolve_status_; }
  const NodeArg* HeadC() const { return head_c_; }

  Status CreateSession(std::unique_ptr<InferenceSession>& session) {
    auto provider = DefaultCudaExecutionProvider();
    if (!provider) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA provider is unavailable");
    std::stringstream stream;
    if (!model_.ToProto().SerializeToOstream(&stream)) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "serialize failed");
    session = std::make_unique<InferenceSession>(SessionOptions{}, GetEnvironment());
    ORT_RETURN_IF_ERROR(session->RegisterExecutionProvider(std::move(provider)));
    ORT_RETURN_IF_ERROR(session->Load(stream));
    return session->Initialize();
  }

 private:
  Model model_;
  Status resolve_status_;
  NodeArg* head_c_ = nullptr;
};

struct FoldInputs {
  TensorShapeVector token_shape;
  std::vector<int64_t> tokens;
  std::vector<int64_t> ids;
  std::vector<int64_t> ctl;
  std::vector<int64_t> context;
  std::vector<int64_t> prefixes;
  std::vector<int64_t> ptrs;
};

struct FoldOutputs {
  std::vector<int32_t> head_c;
  std::vector<int64_t> ctl;
  std::vector<int64_t> context;
};

// Feeds CPU tensors, so ORT copies the state to temporary CUDA tensors: the updated state is only visible
// through the returned outputs.
Status RunFold(InferenceSession& session, const FoldInputs& in, FoldOutputs* out) {
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  NameMLValMap feeds;
  auto feed = [&](const char* name, const TensorShapeVector& shape, const auto& values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    OrtValue value;
    CreateMLValue<T>(allocator, shape, values, &value);
    feeds[name] = std::move(value);
  };
  feed("tokens", in.token_shape, in.tokens);
  feed("ids", {static_cast<int64_t>(in.ids.size())}, in.ids);
  feed("status", {1}, std::vector<int32_t>{0});
  feed("ctl", {static_cast<int64_t>(in.ctl.size())}, in.ctl);
  feed("context", {static_cast<int64_t>(in.context.size())}, in.context);
  feed("prefixes", {static_cast<int64_t>(in.prefixes.size())}, in.prefixes);
  feed("ptrs", {2}, in.ptrs);
  std::vector<OrtValue> outputs;
  const std::vector<std::string> output_names{"head_c", "updated_ctl", "updated_context"};
  ORT_RETURN_IF_ERROR(session.Run(RunOptions{}, feeds, output_names, &outputs));
  auto copy = [](const OrtValue& value, auto* dest) {
    const auto& tensor = value.Get<Tensor>();
    dest->assign(tensor.DataAsSpan<typename std::decay_t<decltype(*dest)>::value_type>().begin(),
                 tensor.DataAsSpan<typename std::decay_t<decltype(*dest)>::value_type>().end());
  };
  copy(outputs[0], &out->head_c);
  copy(outputs[1], &out->ctl);
  copy(outputs[2], &out->context);
  return Status::OK();
}

TEST(SpeculativeFoldTest, SupportsBatchedSpeculativePrefixCommitS8) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80 or newer";
  constexpr int kBatch = 3;
  constexpr int kRows = 8;
  constexpr int kRingCap = 4;
  FoldResultRing ring(kBatch * kRingCap * 8);

  FoldModel model(std::vector<int64_t>{-1, kRows}, kBatch);
  ASSERT_STATUS_OK(model.ResolveStatus());
  const auto* head_c_shape = model.HeadC()->Shape();
  ASSERT_NE(head_c_shape, nullptr);
  ASSERT_EQ(head_c_shape->dim_size(), 1);
  EXPECT_EQ(head_c_shape->dim(0).dim_param(), "batch");
  std::unique_ptr<InferenceSession> session;
  ASSERT_STATUS_OK(model.CreateSession(session));

  FoldInputs in;
  in.token_shape = {kBatch, kRows};
  // Batch 0: accepts 3 drafts (tokens[0..2] == ids[1..3], tokens[3] != ids[4])
  // Batch 1: muted (ctl[3] = 1)
  // Batch 2: accepts 1 draft (tokens[0] == ids[1], tokens[1] != ids[2])
  in.tokens = {10, 11, 12, 99, 14, 15, 16, 17,
               20, 21, 22, 23, 24, 25, 26, 27,
               30, 88, 32, 33, 34, 35, 36, 37};
  in.ids = {1, 10, 11, 12, 13, 14, 15, 16,
            2, 20, 21, 22, 23, 24, 25, 26,
            3, 30, 31, 32, 33, 34, 35, 36};
  in.ctl.assign(kBatch * kCtlWords, 0);
  // Batch 0: length=10, muted=0, seq=0, cache_len=10
  in.ctl[0 * kCtlWords + 1] = 10;
  in.ctl[0 * kCtlWords + 7] = 10;
  // Batch 1: length=20, muted=1, seq=5, cache_len=20
  in.ctl[1 * kCtlWords + 1] = 20;
  in.ctl[1 * kCtlWords + 3] = 1;
  in.ctl[1 * kCtlWords + 4] = 5;
  in.ctl[1 * kCtlWords + 7] = 20;
  // Batch 2: length=30, muted=0, seq=2, cache_len=30
  in.ctl[2 * kCtlWords + 1] = 30;
  in.ctl[2 * kCtlWords + 4] = 2;
  in.ctl[2 * kCtlWords + 7] = 30;
  in.ctl[2 * kCtlWords + 2] = 77;  // untouched word
  in.context = {1, 2, 3, 4, 5, 6};
  in.prefixes.resize(kBatch * kCtlWords);
  for (int b = 0; b < kBatch; ++b) {
    for (int r = 0; r < kRows; ++r) {
      in.prefixes[b * kCtlWords + 2 * r + 0] = 1000 * (b + 1) + 10 * r + 0;
      in.prefixes[b * kCtlWords + 2 * r + 1] = 1000 * (b + 1) + 10 * r + 1;
    }
  }
  in.ptrs = {ring.Address(), kRingCap};

  FoldOutputs out;
  ASSERT_STATUS_OK(RunFold(*session, in, &out));

  EXPECT_EQ(out.head_c, (std::vector<int32_t>{10, 0, 30}));

  EXPECT_EQ(out.ctl[0 * kCtlWords + 0], 99);      // pending = tokens[0, 3]
  EXPECT_EQ(out.ctl[0 * kCtlWords + 1], 10 + 4);  // length = 10 + 3 + 1
  EXPECT_EQ(out.ctl[0 * kCtlWords + 4], 1);       // seq incremented to 1
  EXPECT_EQ(out.ctl[0 * kCtlWords + 6], 3);       // accepted
  EXPECT_EQ(out.ctl[0 * kCtlWords + 7], 10 + 4);  // cache length

  EXPECT_EQ(out.ctl[1 * kCtlWords + 4], 5);  // muted: seq unchanged
  EXPECT_EQ(out.ctl[1 * kCtlWords + 1], 20);

  EXPECT_EQ(out.ctl[2 * kCtlWords + 0], 88);      // pending = tokens[2, 1]
  EXPECT_EQ(out.ctl[2 * kCtlWords + 1], 30 + 2);  // length = 30 + 1 + 1
  EXPECT_EQ(out.ctl[2 * kCtlWords + 2], 77);
  EXPECT_EQ(out.ctl[2 * kCtlWords + 4], 3);  // seq incremented from 2 to 3
  EXPECT_EQ(out.ctl[2 * kCtlWords + 6], 1);

  // Batch 0 accepted=3 -> prefixes[0, 6..7]; batch 1 muted; batch 2 accepted=1 -> prefixes[2, 2..3]
  EXPECT_EQ(out.context, (std::vector<int64_t>{1030, 1031, 3, 4, 3010, 3011}));

  // Ring records published for batch 0 (record 0) and batch 2 (record 2 % kRingCap); batch 1 is untouched.
  const int64_t* host = ring.Host();
  EXPECT_EQ(host[(0 * kRingCap + 0) * 8 + 0], 1);
  for (int word = 0; word < kRingCap * 8; ++word) EXPECT_EQ(host[kRingCap * 8 + word], -1);
  EXPECT_EQ(host[(2 * kRingCap + 2) * 8 + 0], 3);
}

TEST(SpeculativeFoldTest, CarriesUpdatedStateIntoNextRun) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80 or newer";
  constexpr int kRows = 8;
  constexpr int kRingCap = 2;
  FoldResultRing ring(kRingCap * 8);

  FoldModel model(std::vector<int64_t>{1, kRows}, 1);
  ASSERT_STATUS_OK(model.ResolveStatus());
  std::unique_ptr<InferenceSession> session;
  ASSERT_STATUS_OK(model.CreateSession(session));

  FoldInputs in;
  in.token_shape = {1, kRows};
  in.tokens = {10, 11, 12, 13, 14, 15, 16, 17};
  in.ids = {1, 10, 11, 99, 99, 99, 99, 99};  // accepts two drafts
  in.ctl.assign(kCtlWords, 0);
  in.ctl[1] = 10;
  in.ctl[7] = 10;
  in.context = {1, 2};
  in.prefixes.resize(kCtlWords);
  for (int i = 0; i < kCtlWords; ++i) in.prefixes[i] = 100 + i;
  in.ptrs = {ring.Address(), kRingCap};

  for (int run = 1; run <= 3; ++run) {
    SCOPED_TRACE(run);
    FoldOutputs out;
    ASSERT_STATUS_OK(RunFold(*session, in, &out));
    EXPECT_EQ(out.ctl[4], run);  // the sequence counter advances across runs
    EXPECT_EQ(out.ctl[1], 10 + 3 * run);
    EXPECT_EQ(out.ctl[7], 10 + 3 * run);
    EXPECT_EQ(out.head_c[0], 10 + 3 * (run - 1));
    EXPECT_EQ(out.context, (std::vector<int64_t>{104, 105}));
    EXPECT_EQ(ring.Host()[((run - 1) % kRingCap) * 8], run);
    in.ctl = out.ctl;
    in.context = out.context;
  }
  // Three records were published into a two-record ring: record 0 was overwritten by sequence 3, record 1 holds 2.
  EXPECT_EQ(ring.Host()[0], 3);
  EXPECT_EQ(ring.Host()[8], 2);
}

TEST(SpeculativeFoldTest, MutedPrefillAcceptsMoreRowsThanHeadRows) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80 or newer";
  constexpr int kRows = 20;
  FoldResultRing ring(2 * 8);

  FoldModel model(std::vector<int64_t>{kRows}, 1);
  ASSERT_STATUS_OK(model.ResolveStatus());
  const auto* head_c_shape = model.HeadC()->Shape();
  ASSERT_NE(head_c_shape, nullptr);  // a rank-1 next_tokens is batch one
  ASSERT_EQ(head_c_shape->dim_size(), 1);
  EXPECT_EQ(head_c_shape->dim(0).dim_value(), 1);
  std::unique_ptr<InferenceSession> session;
  ASSERT_STATUS_OK(model.CreateSession(session));

  FoldInputs in;
  in.token_shape = {kRows};
  in.tokens.assign(kRows, 10);
  in.ids.assign(kRows, 10);
  in.ctl.assign(kCtlWords, 0);
  in.ctl[1] = 3;
  in.ctl[3] = 1;
  in.ctl[7] = 2;
  in.context = {1, 2};
  in.prefixes.assign(kCtlWords, 100);
  in.ptrs = {0, 0};

  FoldOutputs out;
  ASSERT_STATUS_OK(RunFold(*session, in, &out));
  EXPECT_EQ(out.head_c, std::vector<int32_t>{0});
  EXPECT_EQ(out.ctl, in.ctl);
  EXPECT_EQ(out.context, in.context);
  EXPECT_EQ(std::count(ring.Host(), ring.Host() + ring.Words(), int64_t{-1}), static_cast<int64_t>(ring.Words()));
}

TEST(SpeculativeFoldTest, RejectsNextTokensRankOutsideOneAndTwo) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80 or newer";
  {
    // Schema inference rejects a statically known rank-3 next_tokens.
    FoldModel model(std::vector<int64_t>{2, 3, 8}, 2);
    EXPECT_FALSE(model.ResolveStatus().IsOK());
    EXPECT_THAT(model.ResolveStatus().ErrorMessage(), testing::HasSubstr("next_tokens must have rank 1 or 2"));
  }
  {
    // Without a declared shape, inference declares nothing (no batch 1) and the kernel rejects the runtime shape.
    FoldModel model(std::nullopt, 2);
    ASSERT_STATUS_OK(model.ResolveStatus());
    EXPECT_EQ(model.HeadC()->Shape(), nullptr);
    std::unique_ptr<InferenceSession> session;
    ASSERT_STATUS_OK(model.CreateSession(session));

    FoldInputs in;
    in.token_shape = {2, 3, 8};
    in.tokens.assign(2 * 3 * 8, 10);
    in.ids.assign(2 * 3 * 8, 10);
    in.ctl.assign(2 * kCtlWords, 0);
    in.context = {1, 2, 3, 4};
    in.prefixes.assign(2 * kCtlWords, 100);
    in.ptrs = {0, 0};
    FoldOutputs out;
    const Status status = RunFold(*session, in, &out);
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("next_tokens must have rank 1 ([rows]) or 2 ([batch, rows])"));
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
