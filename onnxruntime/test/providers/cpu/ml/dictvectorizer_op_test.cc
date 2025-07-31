// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MLOpTest, DictVectorizerStringInput) {
  OpTester test("DictVectorizer", 1, onnxruntime::kMLDomain);

  test.AddAttribute("string_vocabulary", std::vector<std::string>{"a", "b", "c", "d"});

  std::map<std::string, int64_t> map;
  map["a"] = 1;
  map["c"] = 2;
  map["d"] = 3;

  test.AddInput<std::string, int64_t>("X", map);

  std::vector<int64_t> dims{1, 4};
  test.AddOutput<int64_t>("Y", dims,
                          {1, 0, 2, 3});
  test.Run();
}

TEST(MLOpTest, DictVectorizerInt64Input) {
  OpTester test("DictVectorizer", 1, onnxruntime::kMLDomain);

  test.AddAttribute("int64_vocabulary", std::vector<int64_t>{1, 2, 3, 4});

  std::map<int64_t, std::string> map;
  map[1] = "a";
  map[3] = "c";
  map[4] = "d";

  test.AddInput<int64_t, std::string>("X", map);

  std::vector<int64_t> dims{1, 4};
  test.AddOutput<std::string>("Y", dims, {"a", "", "c", "d"});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
