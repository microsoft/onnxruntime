// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/../Archive.h"
#include "Featurizers/LabelEncoderFeaturizer.h"
#include "Featurizers/TestHelpers.h"

namespace ft = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

template <typename KeyT, typename IndexT>
using IndexMap = std::unordered_map<KeyT, IndexT>;

namespace {
template <typename InputType>
std::vector<uint8_t> GetStream(const IndexMap<InputType, uint32_t>& map, bool allow_missing_values) {
  ft::Archive ar;
  using TransType = ft::Featurizers::LabelEncoderTransformer<InputType>;
  TransType inst(map, allow_missing_values);
  inst.save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, LabelEncodeTransformer_uint32) {
  OpTester test("LabelEncoderTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  using InputType = uint32_t;

  IndexMap<InputType, uint32_t> index_map = {
      {11, 2}, {8, 0}, {10, 1}, {15, 3}, {20, 5}};

  auto stream = GetStream<InputType>(index_map, false);
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {5}, {11, 8, 10, 15, 20});
  test.AddOutput<uint32_t>("Output", {5}, {2, 0, 1, 3, 5});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, LabelEncodeTransformer_string) {
  OpTester test("LabelEncoderTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  using InputType = std::string;

  IndexMap<InputType, uint32_t> index_map = {
      {"orange", 5}, {"apple", 0}, {"grape", 3}, {"carrot", 5}, {"peach", 5}, {"banana", 1}};

  auto stream = GetStream<InputType>(index_map, false);
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {3}, {"banana", "grape", "apple"});
  test.AddOutput<uint32_t>("Output", {3}, {1, 3, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, LabelEncodeTransformer_string_nothrow) {
  OpTester test("LabelEncoderTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  using InputType = std::string;

  // when an inference data is not seen before, in the non-throw mode, the featurizer should generate 0
  // hello is not seen before among fruits
  IndexMap<InputType, uint32_t> index_map = {
      {"banana", 1},
      {"apple", 2},
      {"grape", 3},
      {"carrot", 4},
      {"peach", 5},
      {"orange", 6}};

  auto stream = GetStream<InputType>(index_map, true);
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {3}, {"banana", "grape", "hello"});
  // The transformer will add 1 to each of the output for the missing input
  test.AddOutput<uint32_t>("Output", {3}, {2, 4, 0});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, LabelEncodeTransformer_string_throw) {
  OpTester test("LabelEncoderTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  using InputType = std::string;

  // when an inference data is not seen before, in the non-throw mode, the featurizer should generate 0
  // hello is not seen before among fruits
  IndexMap<InputType, uint32_t> index_map = {
      {"banana", 1},
      {"apple", 2},
      {"grape", 3},
      {"carrot", 4},
      {"peach", 5},
      {"orange", 6}};

  auto stream = GetStream<InputType>(index_map, false);
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<InputType>("Input", {4}, {"banana", "grape", "apple", "hello"});
  test.AddOutput<uint32_t>("Output", {4}, {1, 3, 2, 0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "'input' was not found");
}

}  // namespace test
}  // namespace onnxruntime
