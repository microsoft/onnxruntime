// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MurmurHash3OpTest, UnsupportedInputType) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<double>("X", {1}, {3.});
  test.AddAttribute<int64_t>("positive", 0);
  test.AddOutput<int32_t>("Y", {1}, {847579505L});
  // Unsupported input type
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(MurmurHash3OpTest, DefaultSeed) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("X", {1}, {3L});
  test.AddAttribute<int64_t>("positive", 0);
  test.AddOutput<int32_t>("Y", {1}, {847579505L});
  test.Run();
}

TEST(MurmurHash3OpTest, ZeroSeed) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("X", {1}, {3L});
  test.AddAttribute<int64_t>("seed", 0LL);
  test.AddAttribute<int64_t>("positive", 0);
  test.AddOutput<int32_t>("Y", {1}, {847579505L});
  test.Run();
}

TEST(MurmurHash3OpTest, ZeroSeedUIntResult) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("X", {1}, {3L});
  test.AddAttribute<int64_t>("seed", 0LL);
  test.AddOutput<uint32_t>("Y", {1}, {847579505L});
  test.Run();
}

TEST(MurmurHash3OpTest, ZeroSeedUIntResult2) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("X", {1}, {4L});
  test.AddAttribute<int64_t>("seed", 0LL);
  test.AddOutput<uint32_t>("Y", {1}, {1889779975L});
  test.Run();
}

TEST(MurmurHash3OpTest, MoreData) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("X", {2}, {3L, 4L});
  test.AddAttribute<int64_t>("seed", 0LL);
  test.AddOutput<uint32_t>("Y", {2}, {847579505L, 1889779975L});
  test.Run();
}

TEST(MurmurHash3OpTest, NonZeroSeed) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("X", {1}, {3L});
  test.AddAttribute<int64_t>("seed", 42LL);
  test.AddAttribute<int64_t>("positive", 0);
  test.AddOutput<int32_t>("Y", {1}, {-1823081949L});
  test.Run();
}

TEST(MurmurHash3OpTest, NonZeroSeedUIntResult) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("X", {1}, {3L});
  test.AddAttribute<int64_t>("seed", 42LL);
  test.AddOutput<uint32_t>("Y", {1}, {2471885347L});
  test.Run();
}

TEST(MurmurHash3OpTest, StringKeyIntResult) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<std::string>("X", {1}, {"foo"});
  test.AddAttribute<int64_t>("seed", 0LL);
  test.AddAttribute<int64_t>("positive", 0);
  test.AddOutput<int32_t>("Y", {1}, {-156908512L});
  test.Run();
}

TEST(MurmurHash3OpTest, StringKeyUIntResult) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<std::string>("X", {1}, {"foo"});
  test.AddAttribute<int64_t>("seed", 0LL);
  test.AddOutput<uint32_t>("Y", {1}, {4138058784L});
  test.Run();
}

TEST(MurmurHash3OpTest, MultipleStringsKeyUIntResult) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<std::string>("X", {2}, {"foo", "bar"});
  test.AddAttribute<int64_t>("seed", 0LL);
  test.AddOutput<uint32_t>("Y", {2}, {4138058784L, 1158584717L});
  test.Run();
}

TEST(MurmurHash3OpTest, StringKeyIntWithSeed42) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<std::string>("X", {1}, {"foo"});
  test.AddAttribute<int64_t>("seed", 42LL);
  test.AddAttribute<int64_t>("positive", 0);
  test.AddOutput<int32_t>("Y", {1}, {-1322301282L});
  test.Run();
}

TEST(MurmurHash3OpTest, StringKeyUIntWithSeed42) {
  OpTester test("MurmurHash3", 1, onnxruntime::kMSDomain);
  test.AddInput<std::string>("X", {1}, {"foo"});
  test.AddAttribute<int64_t>("seed", 42LL);
  test.AddOutput<uint32_t>("Y", {1}, {2972666014L});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
