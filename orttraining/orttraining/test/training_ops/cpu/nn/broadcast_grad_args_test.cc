// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4389)
#endif

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

namespace {
constexpr auto k_opset_version = 1;

const Tensor& FetchTensor(const OrtValue& ort_value) {
  if (ort_value.Fence()) {
    ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, 0);
  }
  return ort_value.Get<Tensor>();
}

void RunBroadcastGradientArgsTest(const char* op, const std::vector<int64_t>& A_shape,
                                  const std::vector<int64_t>& B_shape,
                                  const std::vector<int64_t>& A_axes_true,
                                  const std::vector<int64_t>& B_axes_true) {
  OpTester t{op, k_opset_version, kMSDomain};

  const auto A_size = std::accumulate(
      A_shape.begin(), A_shape.end(), static_cast<int64_t>(1), std::multiplies<>{});
  std::vector<float> A_tensor(A_size);
  std::iota(A_tensor.begin(), A_tensor.end(), 1.0f);
  t.AddInput("a_tensor", A_shape, A_tensor);

  const auto B_size = std::accumulate(
      B_shape.begin(), B_shape.end(), static_cast<int64_t>(1), std::multiplies<>{});
  std::vector<float> B_tensor(B_size);
  std::iota(B_tensor.begin(), B_tensor.end(), 1.0f);
  t.AddInput("b_tensor", B_shape, B_tensor);

  int max_size = int(std::max(A_shape.size(), B_shape.size()));

  std::unique_ptr<int64_t[]> a_axes_buffer{}, b_axes_buffer{};
  a_axes_buffer = onnxruntime::make_unique<int64_t[]>(max_size);
  b_axes_buffer = onnxruntime::make_unique<int64_t[]>(max_size);
  std::vector<int64_t> output_shape = {1, max_size};
  t.AddOutput<int64_t>("a_axes", output_shape, a_axes_buffer.get(), max_size);  // we'll do our own output verification
  t.AddOutput<int64_t>("b_axes", output_shape, b_axes_buffer.get(), max_size);  // we'll do our own output verification

  auto output_verifier = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    ASSERT_GE(fetches.size(), 1);
    const auto& a_axes_op = FetchTensor(fetches[0]);
    auto a_axes_span = a_axes_op.DataAsSpan<int64_t>();
    const auto& b_axes_op = FetchTensor(fetches[1]);
    auto b_axes_span = b_axes_op.DataAsSpan<int64_t>();

    if (A_axes_true.size()) {
      for (auto i = 0; i < A_axes_true.size(); i++) {
        ASSERT_EQ(a_axes_span[i], A_axes_true[i]) << "provider: " << provider_type;
      }
    }
    if (B_axes_true.size()) {
      for (auto i = 0; i < B_axes_true.size(); i++) {
        ASSERT_EQ(b_axes_span[i], B_axes_true[i]) << "provider: " << provider_type;
      }
    }
  };
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  // execution_providers.push_back(DefaultCudaExecutionProvider());
  execution_providers.push_back(DefaultCpuExecutionProvider());
  t.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers, ExecutionMode::ORT_SEQUENTIAL, output_verifier);
}

}  // namespace

// BroadcastGradientArgs

TEST(BroadcastGradientArgsTest, Basic) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 16, 1024, 1024}, {1, 1, 1024, 1024},
                               {}, {1, 0});
}

TEST(BroadcastGradientArgsTest, Basic_1) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 16, 1, 1024}, {1, 1, 1024, 1024},
                               {2}, {1, 0});
}

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime
