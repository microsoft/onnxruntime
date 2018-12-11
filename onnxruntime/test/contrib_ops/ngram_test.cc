// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <stdint.h>

namespace onnxruntime {
namespace test {
namespace ngram_test {

constexpr const char* domain = onnxruntime::kMSDomain;
const int opset_ver = 1;

void InitTestAttr(OpTester& test, const std::string& mode,
                  int64_t M, int64_t N, int64_t S, bool all,
                  const std::vector<int64_t>& ngram_counts,
                  const std::vector<int64_t>& ngram_indexes,
                  const std::vector<float>& weights,
                  const std::vector<int64_t>& pool_int64s,
                  const std::vector<std::string>& pool_strings) {
  test.AddAttribute("mode", mode);
  test.AddAttribute("M", M);
  test.AddAttribute("N", N);
  test.AddAttribute("S", S);
  test.AddAttribute("all", int64_t{all});
  test.AddAttribute("ngram_counts", ngram_counts);
  test.AddAttribute("ngram_indexes", ngram_indexes);
  // optional
  if (!weights.empty()) {
    test.AddAttribute("weights", weights);
  }
  if (!pool_int64s.empty()) {
    test.AddAttribute("pool_int64s", pool_int64s);
  } else {
    test.AddAttribute("pool_strings", pool_strings);
  }
}
}  // namespace ngram_test

using namespace ngram_test;

TEST(ContribOpTest, Ngram_Int32_TF_AllFalse_Skip0) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, int32
  InitTestAttr(test, "TF", 1, 2, 0, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_String_TF_AllFalse_Skip0) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, string
  InitTestAttr(test, "TF", 1, 2, 0, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{2, 6};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_Int32_TF_AllFalse_Skip0_LevelEmpty) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, int32
  InitTestAttr(test, "TF", 1, 2, 0, false,
               {0, 0},
               {
                   0,
                   1,
                   2,
               },  //7 output indexes
               {},
               {                    //1-grams none
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{3};
  std::vector<float> output = {1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_Int32_TF_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=5, , all=false, weights empty, int32
  InitTestAttr(test, "TF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_String_TF_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, string
  InitTestAttr(test, "TF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{2, 6};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_Int32_TF_AllTrue_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=5, , all=false, weights empty, int32
  InitTestAttr(test, "TF", 1, 2, 5, true,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 3, 1, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_String_TF_AllTrue_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, string
  InitTestAttr(test, "TF", 1, 2, 5, true,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{2, 6};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 3, 1, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_Int32_IDF_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=5, , all=false, weights empty, int32
  // We change to IDF but do not supply weights so
  // we should get all 1.0f
  InitTestAttr(test, "IDF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_String_IDF_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, string
  InitTestAttr(test, "IDF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{2, 6};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_Int32_TFIDF_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=5, , all=false, weights empty, int32
  // We change to TFIDF but do not supply weights so
  // we should all get the original values as weights are 1.0f by
  // default
  InitTestAttr(test, "TFIDF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_String_TFIDF_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, string
  InitTestAttr(test, "TFIDF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{2, 6};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_Int32_IDFWeights_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=5, , all=false, weights empty, int32
  // We change to IDF with supplied weights. All
  // with non-zero counts must be replaced with the supplied weights
  InitTestAttr(test, "IDF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 2, 3, 2};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_String_IDFWeights_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, string
  InitTestAttr(test, "IDF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{2, 6};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 2, 3, 2};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_Int32_TFIDFWeights_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=5, , all=false, weights empty, int32
  // We change to TFIDF with supplied weights.
  // We should have all counts scaled by weights
  InitTestAttr(test, "TFIDF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 2, 9, 2};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Ngram_String_TFIDFWeights_AllFalse_Skip5) {
  OpTester test("Ngram", opset_ver, domain);
  // 1 - 2, s=0, , all=false, weights empty, string
  InitTestAttr(test, "TFIDF", 1, 2, 5, false,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{2, 6};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 2, 9, 2};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime
