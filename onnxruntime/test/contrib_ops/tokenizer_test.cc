// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if ((__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L)))
//TODO: handle the u8string.
#else
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace tokenizer_test {
const std::string start_mark{0x2};
const std::string end_mark{0x3};
const std::u8string padval(u8"0xdeadbeaf");

constexpr const char* domain = onnxruntime::kMSDomain;
const int opset_ver = 1;

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
    std::vector<std::string> input{u8"Абсурд", u8"Кома"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Word Absurd is 6 chars long so we must get 6 individual strings out of it
    // which is the max plus start/end text markers
    output_dims.push_back(int64_t(6 + 2));
    std::vector<std::string> output{
        start_mark,
        u8"А", u8"б", u8"с", u8"у", u8"р", u8"д",
        end_mark,
        start_mark,
        u8"К", u8"о", u8"м", u8"а",
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
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Word Absu?? is 6 chars long so we must get 6 individual strings out of it
    // which is the max plus start/end text markers
    output_dims.push_back(int64_t(6 + 2));
    std::vector<std::string> output{
        start_mark,
        u8"А", u8"б", u8"с", u8"у", u8"中", u8"文",
        end_mark,
        start_mark,
        u8"К", u8"о", u8"ñ", u8"ó",
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
    std::vector<std::string> input{u8"", u8""};
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
    std::vector<std::string> input{u8"", u8"", u8"", u8""};
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
    std::string sepexp = u8"(у|ñ)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must split both in 2
    output_dims.push_back(int64_t(2 + 2));
    std::vector<std::string> output{
        start_mark,
        u8"Абс", u8"中文",
        end_mark,
        start_mark,
        u8"Ко", u8"ó",
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }  // namespace test
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersCompleteMatchEmptyOutputC) {
  // Test entire separators match so we get nothing
  // in the output
  {
    std::string sepexp = u8"(Абсу中文)|(Коñó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
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
    std::string sepexp = u8"(А)|(К)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop first characters from both strings
    output_dims.push_back(int64_t(3));
    std::vector<std::string> output{
        start_mark,
        u8"бсу中文",
        end_mark,
        start_mark,
        u8"оñó",
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersEndMatchC) {
  // Match the end
  {
    std::string sepexp = u8"(文)|(ó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop last characters from both strings
    output_dims.push_back(int64_t(3));
    std::vector<std::string> output{
        start_mark,
        u8"Абсу中",
        end_mark,
        start_mark,
        u8"Коñ",
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators_MixCharsWithMarkersEndMatchAtLeast4CharsC) {
  // Match the end, require at least 4 chars
  {
    std::string sepexp = u8"(文)|(ó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 4);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop the last character from the first
    // and the second 3 character token does not pass mincharnum
    output_dims.push_back(int64_t(3));
    std::vector<std::string> output{
        start_mark,
        u8"Абсу中",
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
    std::string sepexp = u8"(文)|(ó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 4);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"", u8""};
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
    std::string sepexp = u8"(文)|(ó)";

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, {sepexp}, 4);

    std::vector<int64_t> dims{2, 2};
    std::vector<std::string> input{u8"", u8"文", u8"ó", u8""};
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
    std::vector<std::string> separators = {u8"су", u8"Абсу"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, separators, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{u8"Абсу中文"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // must split in 2 with no two middle characters
    output_dims.push_back(int64_t(2));
    std::vector<std::string> output{
        u8"Аб", u8"中文"};

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
        u8"Абсу",
        u8"су"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, separators, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{u8"Абсу中文"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop the beginning of the word that
    // also contains the second separator
    output_dims.push_back(int64_t(1));
    std::vector<std::string> output{u8"中文"};

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
        u8"Абсу",
        u8"су"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, separators, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{u8"Абсусусу中文"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop the beginning of the word that
    // also contains the second separator
    output_dims.push_back(int64_t(1));
    std::vector<std::string> output{u8"中文"};

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
        u8"усу",
        u8"Абсу"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, separators, 1);

    std::vector<int64_t> dims{1};
    std::vector<std::string> input{u8"Абсусусу中文"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must drop the beginning of the word that
    // also contains the second separator
    output_dims.push_back(int64_t(2));
    std::vector<std::string> output{u8"Абс", u8"су中文"};

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
      u8";",
      u8";;;"};

  OpTester test("Tokenizer", opset_ver, domain);
  InitTestAttr(test, true, separators, 1);

  std::vector<int64_t> dims{4};
  std::vector<std::string> input{u8"a;b", u8"a;;;b", u8"b;c;;;d;e", u8"a;;b;;;c"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  // Must split both in 2
  output_dims.push_back(int64_t(6));
  std::vector<std::string> output{
      start_mark,
      u8"a",
      u8"b",
      end_mark,
      padval,
      padval,
      start_mark,
      u8"a",
      u8"b",
      end_mark,
      padval,
      padval,
      start_mark,
      u8"b",
      u8"c",
      u8"d",
      u8"e",
      end_mark,
      start_mark,
      u8"a",
      u8"b",
      u8"c",
      end_mark,
      padval,
  };

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}  // namespace test

TEST(ContribOpTest, TokenizerExpression_RegEx) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp(u8"a.");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{4};
  std::vector<std::string> input{u8"a;b", u8"a;;;b", u8"b;c;;;d;e", u8"a;;b;;;c"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(3));
  std::vector<std::string> output{
      start_mark,
      u8"a;",
      end_mark,
      start_mark,
      u8"a;",
      end_mark,
      start_mark,
      end_mark,
      padval,
      start_mark,
      u8"a;",
      end_mark,
  };

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, TokenizerExpression_RegRep) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp(u8"c;+");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{4};
  std::vector<std::string> input{u8"a;b", u8"a;;;b", u8"b;c;;;d;e", u8"a;;b;;;c"};
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
      u8"c;;;",
      end_mark,
      start_mark,
      end_mark,
      padval};

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, TokenizerExpression_Grouping) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp(u8"(a;)|(b;)");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{4};
  std::vector<std::string> input{u8"a;b", u8"a;;;b", u8"b;c;;;d;e", u8"a;;b;;;c"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(4));
  std::vector<std::string> output{
      start_mark,
      u8"a;",
      end_mark,
      padval,
      start_mark,
      u8"a;",
      end_mark,
      padval,
      start_mark,
      u8"b;",
      end_mark,
      padval,
      start_mark,
      u8"a;",
      u8"b;",
      end_mark};

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, TokenizerExpression_RegDot) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp(u8".");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{1};
  std::vector<std::string> input{u8"a;;;b"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(7));
  std::vector<std::string> output{
      start_mark,
      u8"a",
      u8";",
      u8";",
      u8";",
      u8"b",
      end_mark};

  test.AddOutput<std::string>("Y", output_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, TokenizerExpression_RegChar) {
  OpTester test("Tokenizer", opset_ver, domain);
  const std::string tokenexp(u8"\\w");
  InitTestAttr(test, true, {}, 1, tokenexp);

  std::vector<int64_t> dims{1};
  std::vector<std::string> input{u8"a;;;b"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<int64_t> output_dims(dims);
  output_dims.push_back(int64_t(4));
  std::vector<std::string> output{
      start_mark,
      u8"a",
      u8"b",
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
}  // namespace test
}  // namespace onnxruntime
#endif