// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/FromStringFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace dft = NS::Featurizers;

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
std::vector<uint8_t> GetStream() {
  dft::FromStringTransformer<T> transformer;
  NS::Archive ar;
  transformer.save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, FromStringTransformer_bool) {
  using OutputType = bool;
  auto stream = GetStream<OutputType>();
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("FromStringTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddAttribute<int64_t>("result_type", ONNX_NAMESPACE::TensorProto::BOOL);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {3}, {"True", "False", "invalid"});
  test.AddOutput<OutputType>("Output", {3}, {true, false, false});
  test.Run();
}

TEST(FeaturizersTests, FromStringTransformer_int32) {
  using OutputType = int32_t;
  auto stream = GetStream<OutputType>();
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("FromStringTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddAttribute<int64_t>("result_type", ONNX_NAMESPACE::TensorProto::INT32);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {2}, {"10", "-20"});
  test.AddOutput<OutputType>("Output", {2}, {10, -20});
  test.Run();
}

TEST(FeaturizersTests, FromStringTransformer_string) {
  using OutputType = std::string;
  auto stream = GetStream<OutputType>();
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("FromStringTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddAttribute<int64_t>("result_type", ONNX_NAMESPACE::TensorProto::STRING);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {2}, {"10", "-20"});
  test.AddOutput<OutputType>("Output", {2}, {"10", "-20"});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
