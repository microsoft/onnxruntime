// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/ImputationMarkerFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace ft = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {
namespace {
template <typename T>
std::vector<uint8_t> GetStream() {
  ft::Archive ar;
  ft::Featurizers::ImputationMarkerTransformer<T> inst;
  inst.save(ar);
  return ar.commit();
}
}  // namespace

//TEST (FeaturizersTests, ImputationMarker_int8) {
//  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//  auto stream = GetStream<int8_t>();
//  auto dim = static_cast<int64_t>(stream.size());
//
//  test.AddInput<uint8_t>("State", {dim}, stream);
//  test.AddInput<int8_t>("Input", {1}, {25});
//  test.AddOutput<bool>("Output", {1}, {false});
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}
//
//TEST(FeaturizersTests, ImputationMarker_uint8) {
//  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//  auto stream = GetStream<uint8_t>();
//  auto dim = static_cast<int64_t>(stream.size());
//
//  test.AddInput<uint8_t>("State", {dim}, stream);
//  test.AddInput<uint8_t>("Input", {1}, {25});
//  test.AddOutput<bool>("Output", {1}, {false});
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}
//
//TEST(FeaturizersTests, ImputationMarker_int16) {
//  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//  auto stream = GetStream<int16_t>();
//  auto dim = static_cast<int64_t>(stream.size());
//
//  test.AddInput<uint8_t>("State", {dim}, stream);
//  test.AddInput<int16_t>("Input", {1}, {25});
//  test.AddOutput<bool>("Output", {1}, {false});
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}
//
//TEST(FeaturizersTests, ImputationMarker_uint16) {
//  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//  auto stream = GetStream<uint16_t>();
//  auto dim = static_cast<int64_t>(stream.size());
//
//  test.AddInput<uint8_t>("State", {dim}, stream);
//  test.AddInput<uint16_t>("Input", {1}, {25});
//  test.AddOutput<bool>("Output", {1}, {false});
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}
//
//TEST(FeaturizersTests, ImputationMarker_int32) {
//  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//  auto stream = GetStream<int32_t>();
//  auto dim = static_cast<int64_t>(stream.size());
//
//  test.AddInput<uint8_t>("State", {dim}, stream);
//  test.AddInput<int32_t>("Input", {1}, {25});
//  test.AddOutput<bool>("Output", {1}, {false});
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}
//
//TEST(FeaturizersTests, ImputationMarker_uint32) {
//  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//  auto stream = GetStream<uint32_t>();
//  auto dim = static_cast<int64_t>(stream.size());
//
//  test.AddInput<uint8_t>("State", {dim}, stream);
//  test.AddInput<uint32_t>("Input", {1}, {25});
//  test.AddOutput<bool>("Output", {1}, {false});
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}
//
//TEST(FeaturizersTests, ImputationMarker_int64) {
//  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//  auto stream = GetStream<int64_t>();
//  auto dim = static_cast<int64_t>(stream.size());
//
//  test.AddInput<uint8_t>("State", {dim}, stream);
//  test.AddInput<int64_t>("Input", {1}, {25});
//  test.AddOutput<bool>("Output", {1}, {false});
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}
//
//TEST(FeaturizersTests, ImputationMarker_uint64) {
//  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//  auto stream = GetStream<uint64_t>();
//  auto dim = static_cast<int64_t>(stream.size());
//
//  test.AddInput<uint8_t>("State", {dim}, stream);
//  test.AddInput<uint64_t>("Input", {1}, {25});
//  test.AddOutput<bool>("Output", {1}, {false});
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}

TEST(FeaturizersTests, ImputationMarker_float) {
  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto stream = GetStream<float>();
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<float>("Input", {2}, {2.5f, std::numeric_limits<float>::quiet_NaN()});
  test.AddOutput<bool>("Output", {2}, {false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, ImputationMarker_double) {
  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto stream = GetStream<double>();
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<double>("Input", {2}, {2.5, std::numeric_limits<double>::quiet_NaN()});
  test.AddOutput<bool>("Output", {2}, {false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, ImputationMarker_string) {
  OpTester test("ImputationMarkerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  auto stream = GetStream<std::string>();
  auto dim = static_cast<int64_t>(stream.size());

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {2}, {"hello", ""});
  test.AddOutput<bool>("Output", {2}, {false, true});
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}
}
