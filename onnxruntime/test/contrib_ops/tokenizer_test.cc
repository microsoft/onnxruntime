// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace tokenizer_test {
const std::string start_mark{0x2};
const std::string end_mark{0x3};
const std::string padval("0xdeadbeaf");

constexpr const char* domain = onnxruntime::kMSDomain;
constexpr int opset_ver = 1;

}  // namespace tokenizer_test

using namespace tokenizer_test;

void InitTestAttr(OpTester& test, bool mark, const std::vector<std::string>& sepexp,
                  int64_t mincharnum, const std::string& tokenexp = std::string()) {
  test.AddAttribute("mark", int64_t{mark});
  if (!sepexp.empty()) {
    test.AddAttribute("separators", sepexp);
  }

  if (!tokenexp.empty()) {
    test.AddAttribute("tokenexp", tokenexp);
  }
  // Padding for alignment
  test.AddAttribute("pad_value", padval);
  test.AddAttribute("mincharnum", mincharnum);
}

TEST(ContribOpTest, TokenizerCharLevel_LatinCharsNoMarkersC) {
  // Char level tokenezation with latin characters and no
  // start/end text markers
  // [C] dimensions
  // Output [C][D]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {""}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"abcdef", "abcd"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    output_dims.push_back(int64_t(input[0].length()));
    std::vector<std::string> output{
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "a",
        "b",
        "c",
        "d",
        padval,
        padval};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerCharLevel_LatinCharsWithMarkersC) {
  // Char level tokenezation with latin characters and
  // with start/end text markers
  // [C] dimensions
  // Output [C][D]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"abcdef", "abcd"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    output_dims.push_back(int64_t(input[0].length() + 2));
    std::vector<std::string> output{
        start_mark,
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        end_mark,
        start_mark,
        "a",
        "b",
        "c",
        "d",
        end_mark,
        padval,
        padval};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerCharLevel_LatinCharsNoMarkersNC) {
  // Char level tokenezation with latin characters and no
  // start/end text markers
  // [N][C] dimensions
  // Output [N][C][D]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {""}, 1);

    std::vector<int64_t> dims{2, 2};
    std::vector<std::string> input{"abcd", "abcd", "abcd", "abcdef"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    output_dims.push_back(int64_t(input[3].length()));
    std::vector<std::string> output{
        "a",
        "b",
        "c",
        "d",
        padval,
        padval,
        "a",
        "b",
        "c",
        "d",
        padval,
        padval,
        "a",
        "b",
        "c",
        "d",
        padval,
        padval,
        "a",
        "b",
        "c",
        "d",
        "e",
        "f"};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerCharLevel_LatinCharsWithMarkersNC) {
  // Char level tokenezation with latin characters and
  // with start/end text markers
  // [N][C] dimensions
  // Output [N][C][D]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{2, 2};
    std::vector<std::string> input{"abcd", "abcd", "abcd", "abcdef"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    output_dims.push_back(int64_t(input[3].length() + 2));
    std::vector<std::string> output{
        start_mark,
        "a",
        "b",
        "c",
        "d",
        end_mark,
        padval,
        padval,
        start_mark,
        "a",
        "b",
        "c",
        "d",
        end_mark,
        padval,
        padval,
        start_mark,
        "a",
        "b",
        "c",
        "d",
        end_mark,
        padval,
        padval,
        start_mark,
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerCharLevel_CyrillicCharsWithMarkersC) {
  // Char level tokenezation with Cyrillic characters and
  // with start/end text markers
  // [C] dimensions
  // Output [C][D]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"Абсурд", "Кома"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Word Absurd is 6 chars long so we must get 6 individual strings out of it
    // which is the max plus start/end text markers
    output_dims.push_back(int64_t(6 + 2));
    std::vector<std::string> output{
        start_mark,
        "А", "б", "с", "у", "р", "д",
        end_mark,
        start_mark,
        "К", "о", "м", "а",
        end_mark,
        padval,
        padval};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerCharLevel_MixedCharsWithMarkersC) {
  // Char level tokenezation with a mix of latin, Spanish, Cyrillic and Chinese
  // characters and
  // with start/end text markers
  // [C] dimensions
  // Output [C][D]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"Абсу中文", "Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Word Absu?? is 6 chars long so we must get 6 individual strings out of it
    // which is the max plus start/end text markers
    output_dims.push_back(int64_t(6 + 2));
    std::vector<std::string> output{
        start_mark,
        "А", "б", "с", "у", "中", "文",
        end_mark,
        start_mark,
        "К", "о", "ñ", "ó",
        end_mark,
        padval,
        padval};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerCharLevel_EmptyOutputC) {
  // Special case where empty output is produced
  // For [C] we expect [C][0] output
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"", ""};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    output_dims.push_back(int64_t(0));
    std::vector<std::string> output{};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerCharLevel_EmptyOutputNC) {
  // Special case where empty output is produced
  // For [N][C] we expect [N][C][0] output
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{2, 2};
    std::vector<std::string> input{"", "", "", ""};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    output_dims.push_back(int64_t(0));
    std::vector<std::string> output{};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersC) {
  // Separators and strings with a mix of latin, Spanish, Cyrillic and Chinese
  // characters and with start/end text markers
  // [C] dimensions
  // Output [C][D]
  {
    std::string sepexp = "(у|ñ)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"Абсу中文", "Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must split both in 2
    output_dims.push_back(int64_t(2 + 2));
    std::vector<std::string> output{
        start_mark,
        "Абс", "中文",
        end_mark,
        start_mark,
        "Ко", "ó",
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }  // namespace test
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersCompleteMatchEmptyOutputC) {
  // Test entire separators match so we get nothing
  // in the output
  {
    std::string sepexp = "(Абсу中文)|(Коñó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"Абсу中文", "Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must have no output
    output_dims.push_back(int64_t(0));
    std::vector<std::string> output;

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersStartMatchC) {
  // Match the start
  {
    std::string sepexp = "(А)|(К)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"Абсу中文", "Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop first characters from both strings
    output_dims.push_back(int64_t(3));
    std::vector<std::string> output{
        start_mark,
        "бсу中文",
        end_mark,
        start_mark,
        "оñó",
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersEndMatchC) {
  // Match the end
  {
    std::string sepexp = "(文)|(ó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"Абсу中文", "Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop last characters from both strings
    output_dims.push_back(int64_t(3));
    std::vector<std::string> output{
        start_mark,
        "Абсу中",
        end_mark,
        start_mark,
        "Коñ",
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersEndMatchAtLeast4CharsC) {
  // Match the end, require at least 4 chars
  {
    std::string sepexp = "(文)|(ó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 4);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"Абсу中文", "Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop the last character from the first
    // and the second 3 character token does not pass mincharnum
    output_dims.push_back(int64_t(3));
    std::vector<std::string> output{
        start_mark,
        "Абсу中",
        end_mark,
        start_mark,
        end_mark,
        padval};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersEmptyInputEmptyOutputC) {
  // Empty input for [C] should produce [C][0]
  {
    std::string sepexp = "(文)|(ó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 4);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"", ""};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    output_dims.push_back(int64_t(0));
    std::vector<std::string> output;

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}  // namespace test

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersEmptyInputEmptyOutputNC) {
  // Empty input for [N][C] should produce [N][C][0]
  {
    std::string sepexp = "(文)|(ó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 4);

    std::vector<int64_t> dims{2, 2};
    std::vector<std::string> input{"", "文", "ó", ""};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    output_dims.push_back(int64_t(0));
    std::vector<std::string> output;

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapShortFirstC) {
  // Test of the overlapping search patterns
  // The spec mandates that the patterns that appear
  // in the separators earlier must be matched first.
  {
    // In this case the first pattern must match first
    // and there would be no match for the second
    std::vector<std::string> separators = {"су", "Абсу"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, separators, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{"Абсу中文"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // must split in 2 with no two middle characters
    output_dims.push_back(int64_t(2));
    std::vector<std::string> output{
        "Аб", "中文"};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapLongFirstC) {
  // Test of the overlapping search patterns
  // The spec mandates that the patterns that appear
  // in the separators earlier must be matched first.
  {
    // In this case the first pattern must match first
    // and there would be no match for the second
    std::vector<std::string> separators = {
        "Абсу",
        "су"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, separators, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{"Абсу中文"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop the beginning of the word that
    // also contains the second separator
    output_dims.push_back(int64_t(1));
    std::vector<std::string> output{"中文"};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapLongFirstRepeatedShortC) {
  // Test of the overlapping search patterns
  // The spec mandates that the patterns that appear
  // in the separators earlier must be matched first.
  {
    // In this case the first pattern must match first
    // and there would be no match for the second
    std::vector<std::string> separators = {
        "Абсу",
        "су"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, separators, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{"Абсусусу中文"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop the beginning of the word that
    // also contains the second separator
    output_dims.push_back(int64_t(1));
    std::vector<std::string> output{"中文"};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapingMatchC) {
  // Test of the overlapping search patterns
  // The spec mandates that the patterns that appear
  // in the separators earlier must be matched first.
  {
    // In this case the first pattern must match first
    // and there are more than one overlapping matches for the first
    // so the earlier match for the first wins.
    // and there would be no match for the second
    std::vector<std::string> separators = {
        "усу",
        "Абсу"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, separators, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{"Абсусусу中文"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop the beginning of the word that
    // also contains the second separator
    output_dims.push_back(int64_t(2));
    std::vector<std::string> output{"Абс", "су中文"};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharCommonPrefixC) {
  // Separators and strings with a mix of latin, Spanish, Cyrillic and Chinese
  // characters and with start/end text markers
  // [C] dimensions
  // Output [C][D]
  std::vector<std::string> separators = {
      ";",
      ";;;"};

  OpTester test("Tokenizer", opset_ver, domain);
  InitTestAttr(test, true, separators, 1);

  std::vector<int64_t> dims{4};
  std::vector<std::string> input{"a;b", "a;;;b", "b;c;;;d;e", "a;;b;;;c"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  // Must split both in 2
  output_dims.push_back(int64_t(6));
  std::vector<std::string> output{
      start_mark,
      "a",
      "b",
      end_mark,
      padval,
      padval,
      start_mark,
      "a",
      "b",
      end_mark,
      padval,
      padval,
      start_mark,
      "b",
      "c",
      "d",
      "e",
      end_mark,
      start_mark,
      "a",
      "b",
      "c",
      end_mark,
      padval,
  };

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}  // namespace test

TEST(ContribOpTest, TokenizerExpression_RegEx) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp("a.");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{4};
  std::vector<std::string> input{"a;b", "a;;;b", "b;c;;;d;e", "a;;b;;;c"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(3));
  std::vector<std::string> output{
      start_mark,
      "a;",
      end_mark,
      start_mark,
      "a;",
      end_mark,
      start_mark,
      end_mark,
      padval,
      start_mark,
      "a;",
      end_mark,
  };

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, TokenizerExpression_RegRep) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp("c;+");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{4};
  std::vector<std::string> input{"a;b", "a;;;b", "b;c;;;d;e", "a;;b;;;c"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(3));
  std::vector<std::string> output{
      start_mark,
      end_mark,
      padval,
      start_mark,
      end_mark,
      padval,
      start_mark,
      "c;;;",
      end_mark,
      start_mark,
      end_mark,
      padval};

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, TokenizerExpression_Grouping) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp("(a;)|(b;)");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{4};
  std::vector<std::string> input{"a;b", "a;;;b", "b;c;;;d;e", "a;;b;;;c"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(4));
  std::vector<std::string> output{
      start_mark,
      "a;",
      end_mark,
      padval,
      start_mark,
      "a;",
      end_mark,
      padval,
      start_mark,
      "b;",
      end_mark,
      padval,
      start_mark,
      "a;",
      "b;",
      end_mark};

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, TokenizerExpression_RegDot) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp(".");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{1};
  std::vector<std::string> input{"a;;;b"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(7));
  std::vector<std::string> output{
      start_mark,
      "a",
      ";",
      ";",
      ";",
      "b",
      end_mark};

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, TokenizerExpression_RegChar) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp("\\w");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{1};
  std::vector<std::string> input{"a;;;b"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(4));
  std::vector<std::string> output{
      start_mark,
      "a",
      "b",
      end_mark};

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, Tokenizer_EmptyInput) {
  // Special case of empty input.
  // For [C] empty input we should output [0]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{0};
    std::vector<std::string> input;
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    std::vector<std::string> output;

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // For [N][C] empty input we output [N][0]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{1, 0};
    std::vector<std::string> input;
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    std::vector<std::string> output;

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{0, 1};
    std::vector<std::string> input;
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims{0, 0};
    std::vector<std::string> output;

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, Tokenizer_InvalidUtf8Input_CharLevel) {
  // Invalid UTF-8 input should return an error, not crash
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {""}, 1);

    std::vector<int64_t> dims{1};
    // 0xFF is not a valid UTF-8 leading byte
    std::vector<std::string> input{std::string("\xFF\xFE", 2)};
    test.AddInput<std::string>("T", dims, input);

    test.AddOutput<std::string>("Y", {1, 0}, {});

    test.Run(OpTester::ExpectResult::kExpectFailure, "invalid utf8");
  }
}

TEST(ContribOpTest, Tokenizer_InvalidUtf8Input_SeparatorMode) {
  // Invalid UTF-8 input in separator mode should return an error
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {" "}, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{std::string("hello\xFF world", 12)};
    test.AddInput<std::string>("T", dims, input);

    test.AddOutput<std::string>("Y", {1, 0}, {});

    test.Run(OpTester::ExpectResult::kExpectFailure, "invalid utf8");
  }
}

TEST(ContribOpTest, Tokenizer_InvalidUtf8Input_TokenExpMode) {
  // Invalid UTF-8 input in tokenexp mode should return an error
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {}, 1, "\\w+");

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{std::string("hello\xFF world", 12)};
    test.AddInput<std::string>("T", dims, input);

    test.AddOutput<std::string>("Y", {1, 0}, {});

    test.Run(OpTester::ExpectResult::kExpectFailure, "invalid utf8");
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_EmptyMatchRegex) {
  // Regex that can match empty strings (e.g., "a*") should not infinite loop
  // "a*" matches zero or more 'a' chars - can produce empty matches
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {"a*"}, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{"bbb"};
    test.AddInput<std::string>("T", dims, input);

    // "a*" matches empty at every position. The text before each empty match is
    // always 0 characters (< mincharnum=1), so all tokens are filtered out.
    // The advance past empty match consumes each character position.
    std::vector<int64_t> output_dims{1, 0};
    std::vector<std::string> output;

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerExpression_EmptyMatchRegex) {
  // Token expression that can match empty strings - exercises progress guarantee
  {
    OpTester test("Tokenizer", opset_ver, domain);
    // "b?" can match empty or "b" - with longest_match it will match "b" where possible
    const std::string tokenexp("b?");
    InitTestAttr(test, false, {}, 1, tokenexp);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{"abc"};
    test.AddInput<std::string>("T", dims, input);

    // With longest match: matches "b" at position 1
    // Empty matches at other positions are < mincharnum (but mincharnum=1 so empty = 0 chars < 1)
    std::vector<int64_t> output_dims{1, 1};
    std::vector<std::string> output{"b"};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, Tokenizer_EmbeddedNullBytes) {
  // Input with embedded null bytes should be handled correctly
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {" "}, 1);

    std::vector<int64_t> dims{1};
    std::string str_with_null("hello\x00world", 11);
    std::vector<std::string> input{str_with_null};
    test.AddInput<std::string>("T", dims, input);

    // No space separator found, entire string is one token
    std::vector<int64_t> output_dims{1, 1};
    std::vector<std::string> output{str_with_null};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerExpression_MinCharNum) {
  // tokenexp with mincharnum > 1 should filter short matches
  {
    OpTester test("Tokenizer", opset_ver, domain);
    const std::string tokenexp("\\w+");
    InitTestAttr(test, false, {}, 3, tokenexp);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{"I am a developer"};
    test.AddInput<std::string>("T", dims, input);

    // Only "developer" has >= 3 chars. "am" is 2, "I" and "a" are 1.
    std::vector<int64_t> output_dims{1, 1};
    std::vector<std::string> output{"developer"};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerCharLevel_SingleCharWithMark) {
  // Single-character strings with mark=true
  // Output should be [marker, char, marker]
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {""}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{"a", "b"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims{2, 3};
    std::vector<std::string> output{
        start_mark, "a", end_mark,
        start_mark, "b", end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, Tokenizer_LargeInput) {
  // Stress test with larger input to exercise allocation paths
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {" "}, 1);

    constexpr int64_t N = 10;
    constexpr int64_t C = 10;
    std::vector<int64_t> dims{N, C};

    // Create 100 strings, each "word1 word2 word3"
    std::vector<std::string> input;
    input.reserve(N * C);
    for (int i = 0; i < N * C; ++i) {
      input.push_back("hello world foo");
    }
    test.AddInput<std::string>("T", dims, input);

    // Each string splits into 3 tokens + 2 markers = 5
    std::vector<int64_t> output_dims{N, C, 5};
    std::vector<std::string> output;
    output.reserve(N * C * 5);
    for (int i = 0; i < N * C; ++i) {
      output.push_back(start_mark);
      output.push_back("hello");
      output.push_back("world");
      output.push_back("foo");
      output.push_back(end_mark);
    }
    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

}  // namespace test
}  // namespace onnxruntime
