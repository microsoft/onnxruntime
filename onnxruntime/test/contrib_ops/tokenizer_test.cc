// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <codecvt>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace tokenizer_test {
const std::string start_mark{0x2};
const std::string end_mark{0x3};
const std::string padval("0xdeadbeaf");

constexpr const char* domain = onnxruntime::kMSDomain;
const int opset_ver = 1;

}  // namespace tokenizer_test

using namespace tokenizer_test;

void InitTestAttr(OpTester& test, bool mark, const std::vector<std::string>& seps,
                  int64_t mincharnum) {
  test.AddAttribute("mark", int64_t{mark});
  test.AddAttribute("separators", seps);
  // Padding for alignment
  test.AddAttribute("padvalue", padval);
  test.AddAttribute("mincharnum", mincharnum);
}

TEST(ContribOpTest, TokenizerCharLevel) {
  // Mincharnum is too big for character tokenezation, should be 1
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {""}, 2);

    std::vector<int64_t> dims{1, 2};
    std::vector<std::string> input = {std::string("s1"), std::string("s2")};
    test.AddInput<std::string>("T", dims, input);
    std::vector<std::string> output(input);  // do the same for now
    test.AddOutput<std::string>("Y", dims, output);

    test.Run(OpTester::ExpectResult::kExpectFailure, "mincharnum is too big for char level tokenezation");
  }

  // Invalid input dimensions
  {
    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, false, {""}, 1);

    std::vector<int64_t> dims{1, 1, 2};
    std::vector<std::string> input = {std::string("s1"), std::string("s2")};
    test.AddInput<std::string>("T", dims, input);
    std::vector<std::string> output(input);  // do the same for now
    test.AddOutput<std::string>("Y", dims, output);

    test.Run(OpTester::ExpectResult::kExpectFailure, "Input dimensions are either [C] or [N][C] allowed");
  }

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
        padval,
        padval,
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
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
        padval,
        padval,
        end_mark,
        start_mark,
        "a",
        "b",
        "c",
        "d",
        padval,
        padval,
        end_mark,
        start_mark,
        "a",
        "b",
        "c",
        "d",
        padval,
        padval,
        end_mark,
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
        padval,
        padval,
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
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
        padval,
        padval,
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

TEST(ContribOpTest, TokenizerWithSeparators) {
  // Separators and strings with a mix of latin, Spanish, Cyrillic and Chinese
  // characters and with start/end text markers
  // [C] dimensions
  // Output [C][D]
  {
    std::vector<std::string> separators = {
        u8"у",
        u8"ñ"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, separators, 1);

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
  // Test entire separators match so we get nothing
  // in the output
  {
    std::vector<std::string> separators = {
        u8"Абсу中文",
        u8"Коñó"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, separators, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must split both in 2
    output_dims.push_back(int64_t(2));
    std::vector<std::string> output{
        start_mark,
        end_mark,
        start_mark,
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }

  // Match the start
  {
    std::vector<std::string> separators = {
        u8"А",
        u8"К"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, separators, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must split both in 2
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
  // Match the end
  {
    std::vector<std::string> separators = {
        u8"文",
        u8"ó"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, separators, 1);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must split both in 2
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
  // Match the end, require at least 4 chars
  {
    std::vector<std::string> separators = {
        u8"文",
        u8"ó"};

    OpTester test("Tokenizer", opset_ver, domain);
    InitTestAttr(test, true, separators, 4);

    std::vector<int64_t> dims{2};
    std::vector<std::string> input{u8"Абсу中文", u8"Коñó"};
    test.AddInput<std::string>("T", dims, input);

    std::vector<int64_t> output_dims(dims);
    // Must split both in 2
    output_dims.push_back(int64_t(3));
    std::vector<std::string> output{
        start_mark,
        u8"Абсу中",
        end_mark,
        start_mark,
        padval,
        end_mark};

    test.AddOutput<std::string>("Y", output_dims, output);

    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

}  // namespace test
}  // namespace onnxruntime
