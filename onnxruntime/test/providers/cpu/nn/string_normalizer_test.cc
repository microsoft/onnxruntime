// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if ((__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L)))
//TODO: handle the u8string.
#else
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace str_normalizer_test {
constexpr const char* domain = kOnnxDomain;
const int opset_ver = 10;

#ifdef _MSC_VER
const std::string test_locale("en-US");
#else
const std::string test_locale("en_US.UTF-8");
#endif

void InitTestAttr(OpTester& test, const std::string& case_change_action,
                  bool is_case_sensitive,
                  const std::vector<std::string>& stopwords,
                  const std::string& locale) {
  if (!case_change_action.empty()) {
    test.AddAttribute("case_change_action", case_change_action);
  }
  test.AddAttribute("is_case_sensitive", int64_t{is_case_sensitive});
  if (!stopwords.empty()) {
    test.AddAttribute("stopwords", stopwords);
  }
  if (!locale.empty()) {
    test.AddAttribute("locale", locale);
  }
}
}  // namespace str_normalizer_test

using namespace str_normalizer_test;

TEST(ContribOpTest, StringNormalizerTest) {
  // - casesensitive approach
  // - no stopwords.
  // - No change case action, expecting default to take over
  {
    OpTester test("StringNormalizer", opset_ver, domain);
    InitTestAttr(test, "", true, {}, test_locale);
    std::vector<int64_t> dims{4};
    std::vector<std::string> input = {std::string("monday"), std::string("tuesday"),
                                      std::string("wednesday"), std::string("thursday")};
    test.AddInput<std::string>("T", dims, input);
    std::vector<std::string> output(input);  // do the same for now
    test.AddOutput<std::string>("Y", dims, output);
    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // - casesensitive approach
  // - filter out monday
  // - No change case action
  {
    OpTester test("StringNormalizer", opset_ver, domain);
    InitTestAttr(test, "NONE", true, {"monday"}, test_locale);
    std::vector<int64_t> dims{4};
    std::vector<std::string> input = {std::string("monday"), std::string("tuesday"),
                                      std::string("wednesday"), std::string("thursday")};
    test.AddInput<std::string>("T", dims, input);

    std::vector<std::string> output = {std::string("tuesday"),
                                       std::string("wednesday"), std::string("thursday")};
    test.AddOutput<std::string>("Y", {3}, output);
    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // - casesensitive approach
  // - filter out monday
  // - LOWER should produce the same output as they are all lower.
  {
    OpTester test("StringNormalizer", opset_ver, domain);
    InitTestAttr(test, "LOWER", true, {"monday"}, test_locale);
    std::vector<int64_t> dims{4};
    std::vector<std::string> input = {std::string("monday"), std::string("tuesday"),
                                      std::string("wednesday"), std::string("thursday")};
    test.AddInput<std::string>("T", dims, input);

    std::vector<std::string> output = {std::string("tuesday"),
                                       std::string("wednesday"), std::string("thursday")};
    test.AddOutput<std::string>("Y", {3}, output);
    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // - casesensitive approach
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.
  {
    OpTester test("StringNormalizer", opset_ver, domain);
    InitTestAttr(test, "UPPER", true, {"monday"}, test_locale);
    std::vector<int64_t> dims{4};
    std::vector<std::string> input = {std::string("monday"), std::string("tuesday"),
                                      std::string("wednesday"), std::string("thursday")};
    test.AddInput<std::string>("T", dims, input);

    std::vector<std::string> output = {std::string("TUESDAY"),
                                       std::string("WEDNESDAY"), std::string("THURSDAY")};
    test.AddOutput<std::string>("Y", {3}, output);
    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // - case-SENSETIVE approach en_US locale
  // - we test the behavior of a mix of english, french, german, russian and chinese
  //   with en_US locale
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.
  {
    OpTester test("StringNormalizer", opset_ver, domain);
    InitTestAttr(test, "UPPER", true, {u8"monday"}, test_locale);
    std::vector<int64_t> dims{7};
    std::vector<std::string> input = {std::string(u8"monday"),
                                      std::string(u8"tuesday"),
                                      std::string(u8"Besançon"),
                                      std::string(u8"École élémentaire"),
                                      std::string(u8"Понедельник"),
                                      std::string(u8"mit freundlichen grüßen"),
                                      std::string(u8"中文")};
    test.AddInput<std::string>("T", dims, input);

    // en_US results (default)
    std::vector<std::string> output = {std::string(u8"TUESDAY"),
                                       // It does upper case cecedille, accented E
                                       // and german umlaut but fails
                                       // with german eszett
                                       std::string(u8"BESANÇON"),
                                       std::string(u8"ÉCOLE ÉLÉMENTAIRE"),
                                       // No issues with Cyrllic
                                       std::string(u8"ПОНЕДЕЛЬНИК"),
                                       std::string(u8"MIT FREUNDLICHEN GRÜßEN"),
                                       // Chinese do not have cases
                                       std::string(u8"中文")};
    test.AddOutput<std::string>("Y", {6}, output);
    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // - case-INSENSETIVE approach en_US locale
  // - we test the behavior of a mix of english, french, german, russian and chinese
  //   with en_US locale
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.
  {
    OpTester test("StringNormalizer", opset_ver, domain);
    InitTestAttr(test, "UPPER", false, {u8"monday"}, test_locale);
    std::vector<int64_t> dims{7};
    std::vector<std::string> input = {std::string(u8"monday"),
                                      std::string(u8"tuesday"),
                                      std::string(u8"Besançon"),
                                      std::string(u8"École élémentaire"),
                                      std::string(u8"Понедельник"),
                                      std::string(u8"mit freundlichen grüßen"),
                                      std::string(u8"中文")};
    test.AddInput<std::string>("T", dims, input);

    // en_US results (default)
    std::vector<std::string> output = {std::string(u8"TUESDAY"),
                                       // It does upper case cecedille, accented E
                                       // and german umlaut but fails
                                       // with german eszett
                                       std::string(u8"BESANÇON"),
                                       std::string(u8"ÉCOLE ÉLÉMENTAIRE"),
                                       // No issues with Cyrllic
                                       std::string(u8"ПОНЕДЕЛЬНИК"),
                                       std::string(u8"MIT FREUNDLICHEN GRÜßEN"),
                                       // Chinese do not have cases
                                       std::string(u8"中文")};
    test.AddOutput<std::string>("Y", {6}, output);
    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }

  // Empty output case
  // - casesensitive approach
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.
  {
    OpTester test("StringNormalizer", opset_ver, domain);
    InitTestAttr(test, "UPPER", true, {"monday"}, test_locale);
    std::vector<int64_t> dims{2};
    std::vector<std::string> input = {std::string("monday"),
                                      std::string("monday")};
    test.AddInput<std::string>("T", dims, input);

    std::vector<std::string> output{""};  // One empty string
    test.AddOutput<std::string>("Y", {1}, output);
    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // Empty output case
  // - casesensitive approach
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.
  {
    OpTester test("StringNormalizer", opset_ver, domain);
    InitTestAttr(test, "UPPER", true, {"monday"}, "");
    std::vector<int64_t> dims{1, 2};
    std::vector<std::string> input = {std::string("monday"),
                                      std::string("monday")};
    test.AddInput<std::string>("T", dims, input);

    std::vector<std::string> output{""};  // One empty string
    test.AddOutput<std::string>("Y", {1, 1}, output);
    test.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

}  // namespace test
}  // namespace onnxruntime
#endif