// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(BifurcationDetectorTest, Test1) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {1}, {2});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddInput<int64_t>("pred_tokens", {5}, {1, 5, 3, 4, 2});
  tester.AddOutput<int64_t>("tokens", {6}, {2, 1, 5, 3, 4, 2});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(BifurcationDetectorTest, Test2) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("src_tokens", {26}, {756, 194, 39, 1015, 5529, 1216, 24, 72, 23, 1976, 6174, 1340, 6, 39, 194, 2161, 1480, 4955, 8, 7806, 65, 1091, 8, 560, 4077, 196});
  tester.AddInput<int64_t>("cur_tokens", {6}, {2, 756, 194, 39, 8155, 23});
  tester.AddInput<int64_t>("find_end_idx", {}, {0});
  tester.AddOutput<int64_t>("tokens", {6}, {2, 756, 194, 39, 8155, 23});
  tester.AddOutput<int64_t>("new_end_idx", {}, {9});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Bifurcation at the first predicted token (immediate mismatch).
// pred_tokens[0] != src_tokens[prev_suffix_match_idx] → pred_bifur_idx = 0.
// Output = cur_tokens + pred_tokens[0].
TEST(BifurcationDetectorTest, BifurcationAtFirstToken) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src=[1,2,3], prev_idx=0, pred=[99,2,3,0] (pred[0]=99 != src[0]=1).
  tester.AddInput<int64_t>("src_tokens", {3}, {1, 2, 3});
  tester.AddInput<int64_t>("cur_tokens", {2}, {10, 20});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddInput<int64_t>("pred_tokens", {4}, {99, 2, 3, 0});
  // pred_bifur_idx = 0, output = [10, 20] + [99] = [10, 20, 99]
  tester.AddOutput<int64_t>("tokens", {3}, {10, 20, 99});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Bifurcation in the middle of the predicted sequence.
TEST(BifurcationDetectorTest, BifurcationMidSequence) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src=[10,20,30,40], prev_idx=0, pred=[10,20,99,40,0].
  // Match at pred[0]=10==src[0], pred[1]=20==src[1], pred[2]=99!=src[2]=30.
  // pred_bifur_idx = 2. Output = cur + pred[0..2] = [5] + [10,20,99].
  tester.AddInput<int64_t>("src_tokens", {4}, {10, 20, 30, 40});
  tester.AddInput<int64_t>("cur_tokens", {1}, {5});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddInput<int64_t>("pred_tokens", {5}, {10, 20, 99, 40, 0});
  tester.AddOutput<int64_t>("tokens", {4}, {5, 10, 20, 99});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Non-zero prev_suffix_match_idx with pred_tokens: bifurcation scan starts
// partway through src_tokens.
TEST(BifurcationDetectorTest, NonZeroPrevSuffixMatchIdx) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src=[10,20,30,40,50], prev_idx=2.
  // pred_tokens_len must be 5 + 1 - 2 = 4.
  // Compare: pred[0] vs src[2]=30, pred[1] vs src[3]=40, pred[2] vs src[4]=50.
  // pred=[30,40,99,0] → match at 0,1; mismatch at 2. pred_bifur_idx=2.
  // Output = [5] + [30,40,99].
  tester.AddInput<int64_t>("src_tokens", {5}, {10, 20, 30, 40, 50});
  tester.AddInput<int64_t>("cur_tokens", {1}, {5});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {2});
  tester.AddInput<int64_t>("pred_tokens", {4}, {30, 40, 99, 0});
  tester.AddOutput<int64_t>("tokens", {4}, {5, 30, 40, 99});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Suffix matching: multiple occurrences of the 1-gram cause suffix_idx = -1,
// then the 2-gram is unique → suffix_idx reports the 2-gram match position.
TEST(BifurcationDetectorTest, SuffixMatchMultipleSingleGramUniqueDigram) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src=[1,3,4,2,1,4,0], cur=[5,1,4]. No pred → output = [5,1,4].
  // 1-gram [4]: found at src[2] and src[5] → multiple → -1, continue.
  // 2-gram [1,4]: found at src[4..5]. suffix_idx=4+2=6. No second match → unique.
  tester.AddInput<int64_t>("src_tokens", {7}, {1, 3, 4, 2, 1, 4, 0});
  tester.AddInput<int64_t>("cur_tokens", {3}, {5, 1, 4});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  // No pred_tokens → output = cur_tokens = [5, 1, 4].
  tester.AddOutput<int64_t>("tokens", {3}, {5, 1, 4});
  // 1-gram [4]: multiple matches → -1, continue.
  // 2-gram [1,4]: unique match at src[4..5], suffix_idx = 4+2 = 6.
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {6});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Suffix matching: suffix_idx >= src_tokens_len causes an early break after assignment,
// so this edge case returns the assigned suffix_idx, not -1.
TEST(BifurcationDetectorTest, SuffixMatchAtEndOfSrc) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src=[1,2,3], cur=[5,3].
  // 1-gram: [3]. Found at src[2]. suffix_idx = 2+1 = 3 >= 3 → break.
  // suffix_idx was already assigned 3 before the break, so the result is 3.
  tester.AddInput<int64_t>("src_tokens", {3}, {1, 2, 3});
  tester.AddInput<int64_t>("cur_tokens", {2}, {5, 3});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddOutput<int64_t>("tokens", {2}, {5, 3});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {3});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Suffix matching: n-gram size exceeds output token count → early break.
TEST(BifurcationDetectorTest, SuffixNgramExceedsOutputLen) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("min_ngram_size", int64_t(5));
  tester.AddAttribute<int64_t>("max_ngram_size", int64_t(7));

  // Output has only 2 tokens, but min_ngram_size=5. The loop immediately breaks
  // because i=5 > tokens_len=2. suffix_idx stays -1.
  tester.AddInput<int64_t>("src_tokens", {10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  tester.AddInput<int64_t>("cur_tokens", {2}, {5, 3});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddOutput<int64_t>("tokens", {2}, {5, 3});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Custom min/max_ngram_size: min=2, max=2. Only 2-grams are checked.
TEST(BifurcationDetectorTest, CustomNgramSize) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("min_ngram_size", int64_t(2));
  tester.AddAttribute<int64_t>("max_ngram_size", int64_t(2));

  // src=[1,2,3,4,5], cur=[7,3,4].
  // With default min=1: 1-gram [4] found at src[3], suffix_idx=4, unique → return 4.
  // With min=max=2: only 2-gram [3,4] is checked. Found at src[2..3], suffix_idx=2+2=4. unique → return 4.
  // Same result here but exercises the attribute path.
  tester.AddInput<int64_t>("src_tokens", {5}, {1, 2, 3, 4, 5});
  tester.AddInput<int64_t>("cur_tokens", {3}, {7, 3, 4});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddOutput<int64_t>("tokens", {3}, {7, 3, 4});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {4});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Combined: non-zero prev_suffix_match_idx with pred_tokens AND suffix match.
// Exercises both major code paths together.
TEST(BifurcationDetectorTest, BifurcationAndSuffixMatchCombined) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src=[10,20,30,40,50,60], prev_idx=3.
  // pred_tokens_len = 6 + 1 - 3 = 4.
  // Compare pred vs src starting at offset 3: pred[0] vs src[3]=40, pred[1] vs src[4]=50, pred[2] vs src[5]=60.
  // pred=[40,50,99,0]. Match at 0,1; mismatch at 2. pred_bifur_idx=2.
  // Output = cur + pred[0..2] = [5, 8] + [40, 50, 99] = [5, 8, 40, 50, 99].
  //
  // Suffix matching on output=[5,8,40,50,99] against src=[10,20,30,40,50,60]:
  // 1-gram: [99]. Not in src → break. suffix_idx=-1.
  tester.AddInput<int64_t>("src_tokens", {6}, {10, 20, 30, 40, 50, 60});
  tester.AddInput<int64_t>("cur_tokens", {2}, {5, 8});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {3});
  tester.AddInput<int64_t>("pred_tokens", {4}, {40, 50, 99, 0});
  tester.AddOutput<int64_t>("tokens", {5}, {5, 8, 40, 50, 99});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Verify that a negative prev_suffix_match_idx is rejected.
TEST(BifurcationDetectorTest, NegativePrevSuffixMatchIdx) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src_tokens has 4 elements. With prev_suffix_match_idx = -1,
  // pred_tokens_len must satisfy: pred_tokens_len == src_tokens_len + 1 - (-1) = 6
  // The negative index must be caught before it is used as an array offset.
  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {1}, {2});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {-1});
  tester.AddInput<int64_t>("pred_tokens", {6}, {1, 5, 3, 4, 2, 7});
  tester.AddOutput<int64_t>("tokens", {1}, {0});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {0});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "prev_suffix_match_idx must be non-negative",
             {}, nullptr, &execution_providers);
}

// Verify that a large negative prev_suffix_match_idx is also rejected.
TEST(BifurcationDetectorTest, LargeNegativePrevSuffixMatchIdx) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {1}, {2});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {-100});
  tester.AddInput<int64_t>("pred_tokens", {105}, std::vector<int64_t>(105, 0));
  tester.AddOutput<int64_t>("tokens", {1}, {0});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {0});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "prev_suffix_match_idx must be non-negative",
             {}, nullptr, &execution_providers);
}

// Verify prev_suffix_match_idx exceeding src_tokens_len is rejected.
TEST(BifurcationDetectorTest, PrevSuffixMatchIdxExceedsSrcLen) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src_tokens_len = 4, prev_suffix_match_idx = 5 should fail the upper-bound check.
  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {1}, {2});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {5});
  tester.AddInput<int64_t>("pred_tokens", {1}, {7});
  tester.AddOutput<int64_t>("tokens", {1}, {0});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {0});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "prev_suffix_match_idx must not exceed src_tokens length",
             {}, nullptr, &execution_providers);
}

// No pred_tokens — output should equal cur_tokens.
TEST(BifurcationDetectorTest, NoPredTokens) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {3}, {10, 20, 30});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddOutput<int64_t>("tokens", {3}, {10, 20, 30});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// prev_suffix_match_idx at the boundary (equal to src_tokens_len).
TEST(BifurcationDetectorTest, PrevSuffixMatchIdxAtBoundary) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // prev_suffix_match_idx = 4 = src_tokens_len.
  // pred_tokens_len must be src_tokens_len + 1 - 4 = 1.
  // Loop doesn't execute (bound = 0), pred_bifur_idx = 0.
  // Output = cur_tokens + pred_tokens[0..0].
  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {2}, {10, 20});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {4});
  tester.AddInput<int64_t>("pred_tokens", {1}, {99});
  tester.AddOutput<int64_t>("tokens", {3}, {10, 20, 99});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// All predicted tokens match source tokens — no bifurcation occurs.
// pred_bifur_idx reaches the loop bound (src_tokens_len - prev_suffix_match_idx).
// Output = cur_tokens + all pred_tokens.
TEST(BifurcationDetectorTest, FullMatchNoBifurcation) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src=[10,20,30], prev_idx=0, pred must have len = 3+1-0 = 4.
  // pred=[10,20,30,99]. Loop: pred[0]==src[0], pred[1]==src[1], pred[2]==src[2].
  // pred_bifur_idx = 3 (loop bound). Output = [5] + pred[0..3] = [5,10,20,30,99].
  tester.AddInput<int64_t>("src_tokens", {3}, {10, 20, 30});
  tester.AddInput<int64_t>("cur_tokens", {1}, {5});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddInput<int64_t>("pred_tokens", {4}, {10, 20, 30, 99});
  tester.AddOutput<int64_t>("tokens", {5}, {5, 10, 20, 30, 99});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {-1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// pred_tokens length does not match the expected (src_tokens_len + 1 - prev_suffix_match_idx).
TEST(BifurcationDetectorTest, PredTokensLengthMismatch) {
  OpTester tester("BifurcationDetector", 1, onnxruntime::kMSDomain);

  // src_tokens_len=4, prev_suffix_match_idx=0 → expected pred_tokens_len = 5.
  // Provide pred_tokens_len = 3 to trigger the mismatch check.
  tester.AddInput<int64_t>("src_tokens", {4}, {1, 5, 3, 4});
  tester.AddInput<int64_t>("cur_tokens", {1}, {2});
  tester.AddInput<int64_t>("prev_suffix_match_idx", {}, {0});
  tester.AddInput<int64_t>("pred_tokens", {3}, {1, 5, 3});
  tester.AddOutput<int64_t>("tokens", {1}, {0});
  tester.AddOutput<int64_t>("suffix_match_idx", {}, {0});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "pred_tokens length mismatch",
             {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
