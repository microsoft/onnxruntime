// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <stdint.h>

namespace onnxruntime {
namespace test {
namespace tfidfvectorizer_test {

constexpr const char* domain = kOnnxDomain;
const int opset_ver = 9;

void InitTestAttr(OpTester& test, const std::string& mode,
                  int64_t min_gram_length, int64_t max_gram_length, int64_t max_skip_count,
                  const std::vector<int64_t>& ngram_counts,
                  const std::vector<int64_t>& ngram_indexes,
                  const std::vector<float>& weights,
                  const std::vector<int64_t>& pool_int64s,
                  const std::vector<std::string>& pool_strings) {
  test.AddAttribute("mode", mode);
  test.AddAttribute("min_gram_length", min_gram_length);
  test.AddAttribute("max_gram_length", max_gram_length);
  test.AddAttribute("max_skip_count", max_skip_count);
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
}  // namespace tfidfvectorizer_test

using namespace tfidfvectorizer_test;

// Here is what takes place in general and in particular
// in this unit test.There are 7 n - grams : 4 unigrams and 3 bigrams
// that are expressed as 10 items(integers in this case) contained within pool_int64 attribute.
// We only count and then optionally scale those ngrams that appear in the supplied pool parameter(either int64 or string).
// M = 1 and N = 2 in this case.
// However, attribute all controls whether we consider all of the supplied ngram[M..N] sizes
// into consideration or not.With all = false, we only consider N - grams.

TEST(TfIdfVectorizerTest, Int32_TF_onlyBigrams_Skip0) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{12};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TF_onlyBigrams_Skip0_Empty_Dim1Fail) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{0};
  std::vector<int32_t> input = {};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{0};
  std::vector<float> output = {};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(TfIdfVectorizerTest, Int32_TF_onlyBigrams_Skip0_Empty_Dim1Success) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{0};
  std::vector<int32_t> input = {};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 0, 0, 0};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TF_onlyBigrams_Skip0_Empty_Dim2) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{1, 0};
  std::vector<int32_t> input = {};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 0, 0, 0};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(TfIdfVectorizerTest, Int32_TF_onlyBigrams_Skip01_Empty_Dim2) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{0, 1};
  std::vector<int32_t> input = {};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 0, 0, 0};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(TfIdfVectorizerTest, Int32_TF_onlyBigrams_Skip0_Empty_Dim2N) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 0};
  std::vector<int32_t> input = {};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{2, 7};
  std::vector<float> output = {0, 0, 0, 0, 0, 0, 0, 
                               0, 0, 0, 0, 0, 0, 0};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TF_BatchOnlyBigrams_Skip0) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  // Tow batches by six
  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7,
                                8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{2, 7};
  std::vector<float> output = {0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 1, 0, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_TF_OnlyBigrams_Skip0) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, string
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{12};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_TF_BatchOnlyBigrams_Skip0) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, string
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{2, 6};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven",
                                 "eight", "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{2, 7};
  // ["seven", "eight"] can not be found due to batch boundary and s=0
  // bigram elements have to be next to each other
  std::vector<float> output = {0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 1, 0, 1};

  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TF_onlyBigrams_LevelEmpty) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=0, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 0,
               {0, 0},  // no unigrams, bi-grams start immediately
               {
                   0,
                   1,
                   2,
               },  //7 output indexes
               {},
               {                    //1-grams none
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{12};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{3};
  std::vector<float> output = {1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TF_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{12};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  // No 1-grams but Skip is 5 so we manage to count 3
  // occurrences of [7,8]
  std::vector<float> output = {0, 0, 0, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TF_BatchOnlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, , Min=Max=2, weights empty, int32
  InitTestAttr(test, "TF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7,
                                8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{2, 7};
  // Skip is 5 but we are constraint by row boundaries
  // so count only 1 of each
  std::vector<float> output = {0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_TF_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, , Min=Max=2, weights empty, string
  InitTestAttr(test, "TF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{12};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  // No 1-grams but Skip is 5 so we manage to count 3
  // occurrences of [7,8] in one batch (row)
  std::vector<float> output = {0, 0, 0, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_TF_BatchOnlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, , Min=Max=2, weights empty, string
  InitTestAttr(test, "TF", 2, 2, 5,
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

  std::vector<int64_t> out_dims{2, 7};
  std::vector<float> output = {0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TF_UniAndBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, , Min=1, Max=2, weights empty, int32
  InitTestAttr(test, "TF", 1, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{12};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  // We consider both 1-grams and 2-grams so get all the counts here
  std::vector<float> output = {0, 3, 1, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TF_BatchUniAndBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=1, Max=2, weights empty, int32
  InitTestAttr(test, "TF", 1, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{2, 6};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7,
                                8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{2, 7};
  // Counts are now per row (batch)
  std::vector<float> output = {0, 3, 0, 0, 0, 0, 0,
                               0, 0, 1, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_TF_UniAndBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=1, Max=2, weights empty, string
  InitTestAttr(test, "TF", 1, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{12};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 3, 1, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_TF_BatchUniAndBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=1, Max=2, weights empty, string
  InitTestAttr(test, "TF", 1, 2, 5,
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

  std::vector<int64_t> out_dims{2, 7};
  std::vector<float> output = {0, 3, 0, 0, 0, 0, 0,
                               0, 0, 1, 0, 1, 1, 1};

  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_IDF_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights empty, int32
  // We change to IDF but do not supply weights so
  // we should get all 1.0f where count is not zero
  InitTestAttr(test, "IDF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{12};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_IDF_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights empty, string
  InitTestAttr(test, "IDF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{12};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 1, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TFIDF_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights empty, int32
  // We change to TFIDF but do not supply weights so
  // we should all get the original values as weights are 1.0f by
  // default
  InitTestAttr(test, "TFIDF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {2, 3, 5, 4,         //1-grams
                5, 6, 7, 8, 6, 7},  //bi-grams
               {});

  std::vector<int64_t> dims{12};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_TFIDF_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights empty, string
  InitTestAttr(test, "TFIDF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},  //7 output indexes
               {},
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{12};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 1, 3, 1};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_IDFWeights_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights specified, int32
  // We change to IDF with supplied weights. All
  // with non-zero counts must be replaced with the supplied weights
  InitTestAttr(test, "IDF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},                //7 output indexes
               {2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0},  // weights
               {2, 3, 5, 4,                          //1-grams
                5, 6, 7, 8, 6, 7},                   //bi-grams
               {});

  std::vector<int64_t> dims{12};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 2, 3, 2};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_IDFWeights_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights specified, string
  InitTestAttr(test, "IDF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},                //7 output indexes
               {2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0},  // weights
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{12};
  std::vector<std::string> input{"one", "one", "three", "three", "three", "seven", "eight",
                                 "six", "seven", "five", "six", "eight"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 2, 3, 2};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, Int32_TFIDFWeights_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights specified, int32
  // We change to TFIDF with supplied weights.
  // We should have all counts scaled by weights
  InitTestAttr(test, "TFIDF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},                //7 output indexes
               {2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0},  // weights
               {2, 3, 5, 4,                          //1-grams
                5, 6, 7, 8, 6, 7},                   //bi-grams
               {});

  std::vector<int64_t> dims{12};
  std::vector<int32_t> input = {1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8};
  test.AddInput<int32_t>("T", dims, input);

  std::vector<int64_t> out_dims{7};
  std::vector<float> output = {0, 0, 0, 0, 2, 9, 2};
  test.AddOutput<float>("Y", out_dims, output);

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(TfIdfVectorizerTest, String_TFIDFWeights_onlyBigrams_Skip5) {
  OpTester test("TfIdfVectorizer", opset_ver, domain);
  // s=5, Min=Max=2, weights specified, string
  InitTestAttr(test, "TFIDF", 2, 2, 5,
               {0, 4},
               {0, 1, 2, 3, 4, 5, 6},                //7 output indexes
               {2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0},  // weights
               {},
               {"two", "three", "five", "four",                     //1-grams
                "five", "six", "seven", "eight", "six", "seven"});  //bi-grams

  std::vector<int64_t> dims{12};
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
