// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IOS
#define ORT_IOS
#endif
#endif

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

static void InitTestAttr(OpTester& test, const std::string& case_change_action,
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

TEST(ContribOpTest, StringNormalizerSensitiveNoCase) {
  // - casesensitive approach
  // - no stopwords.
  // - No change case action, expecting default to take over
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

TEST(ContribOpTest, StringNormalizerSensitiveFilterOutNoCase) {
  // - casesensitive approach
  // - filter out monday
  // - No change case action

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

TEST(ContribOpTest, StringNormalizerSensitiveFilterOutLower) {
  // - casesensitive approach
  // - filter out monday
  // - LOWER should produce the same output as they are all lower.
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

TEST(ContribOpTest, StringNormalizerSensitiveFilterOutUpper) {
  // - casesensitive approach
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.

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

TEST(ContribOpTest, StringNormalizerSensitiveFilterOutUpperWithLocale) {
  // - case-SENSITIVE approach en_US locale
  // - we test the behavior of a mix of english, french, german, russian and chinese
  //   with en_US locale
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.

  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "UPPER", true, {"monday"}, test_locale);
  std::vector<int64_t> dims{7};
  std::vector<std::string> input = {"monday",
                                    "tuesday",
                                    "Besançon",
                                    "École élémentaire",
                                    "Понедельник",
                                    "mit freundlichen grüßen",
                                    "中文"};
  test.AddInput<std::string>("T", dims, input);

  // en_US results (default)
  std::vector<std::string> output = {"TUESDAY",
                                     // It does upper case cecedille, accented E
                                     "BESANÇON",
                                     "ÉCOLE ÉLÉMENTAIRE",
                                     // No issues with Cyrllic
                                     "ПОНЕДЕЛЬНИК",
  // Works with german umlaut but fails
  // with german Eszett. Reason being, capital case for Eszett
  // was introduced only recently into encodings
  // and some platforms produce it, but others do not
#ifdef __wasm__
                                     "MIT FREUNDLICHEN GRÜẞEN",
#else
                                     "MIT FREUNDLICHEN GRÜßEN",
#endif
                                     // Chinese do not have cases
                                     "中文"};
  test.AddOutput<std::string>("Y", {6}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerInsensitiveFilterOutUpperWithLocale) {
  // - case-INSENSITIVE approach en_US locale
  // - we test the behavior of a mix of english, french, german, russian and chinese
  //   with en_US locale
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.

  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "UPPER", false, {"monday"}, test_locale);
  std::vector<int64_t> dims{7};
  std::vector<std::string> input = {"monday",
                                    "tuesday",
                                    "Besançon",
                                    "École élémentaire",
                                    "Понедельник",
                                    "mit freundlichen grüßen",
                                    "中文"};
  test.AddInput<std::string>("T", dims, input);

  // en_US results (default)
  std::vector<std::string> output = {"TUESDAY",
                                     // It does upper case cecedille, and accented E
                                     "BESANÇON",
                                     "ÉCOLE ÉLÉMENTAIRE",
                                     // No issues with Cyrllic
                                     "ПОНЕДЕЛЬНИК",
  // Works with german umlaut but fails
  // with german Eszett (ß). Reason being, capital case for Eszett
  // was introduced only recently into encodings
  // and some platforms produce it, but others do not
#ifdef __wasm__
                                     "MIT FREUNDLICHEN GRÜẞEN",
#else
                                     "MIT FREUNDLICHEN GRÜßEN",
#endif

                                     // Chinese do not have cases
                                     "中文"};
  test.AddOutput<std::string>("Y", {6}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerSensitiveFilterOutUpperEmptyCase) {
  // Empty output case
  // - casesensitive approach
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.

  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "UPPER", true, {"monday"}, test_locale);
  std::vector<int64_t> dims{2};
  std::vector<std::string> input = {"monday", "monday"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output{""};  // One empty string
  test.AddOutput<std::string>("Y", {1}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

// Fails on iOS because necessary locales are not installed
// MacOS runs fine.
#ifndef ORT_IOS
TEST(ContribOpTest, StringNormalizerSensitiveFilterOutUpperSameOutput) {
  // Empty output case
  // - casesensitive approach
  // - filter out monday
  // - UPPER should produce the same output as they are all lower.
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
#endif

// ============================================================
// Additional tests for coverage gaps
// ============================================================

TEST(ContribOpTest, StringNormalizerInsensitiveFilterOutLower) {
  // Case-insensitive filtering + LOWER case change.
  // This exercises the can_reuse_wide fast path (compare_caseaction_ == LOWER == case_change_action_).
  // Tests French (accented), German (umlaut/eszett), Russian, Chinese.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "LOWER", false, {"Понедельник", "Besançon"}, test_locale);
  std::vector<int64_t> dims{6};
  std::vector<std::string> input = {"ПОНЕДЕЛЬНИК",  // matches "Понедельник" case-insensitively
                                    "BESANÇON",     // matches "Besançon" case-insensitively
                                    "École élémentaire",
                                    "mit freundlichen grüßen",
                                    "中文",
                                    "Tuesday"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"école élémentaire",
                                     "mit freundlichen grüßen",  // ß stays ß when lowercased
                                     "中文",                     // Chinese has no case
                                     "tuesday"};
  test.AddOutput<std::string>("Y", {4}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerInsensitiveFilterOutNone) {
  // Case-insensitive filtering + NO case change.
  // Strings matching stopwords are removed; survivors keep original case.
  // Tests that Cyrillic and accented Latin stopwords match case-insensitively.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "NONE", false, {"понедельник", "école élémentaire"}, test_locale);
  std::vector<int64_t> dims{5};
  std::vector<std::string> input = {"Понедельник",        // matches "понедельник"
                                    "École Élémentaire",  // matches "école élémentaire"
                                    "Besançon",
                                    "中文",
                                    "Thursday"};
  test.AddInput<std::string>("T", dims, input);

  // Filtered strings are removed; survivors keep original case
  std::vector<std::string> output = {"Besançon", "中文", "Thursday"};
  test.AddOutput<std::string>("Y", {3}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerInsensitiveNoStopwordsLower) {
  // Case-insensitive, no stopwords, LOWER case change.
  // Exercises output_no_filtering path with multilingual input.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "LOWER", false, {}, test_locale);
  std::vector<int64_t> dims{5};
  std::vector<std::string> input = {"BESANÇON",
                                    "ÉCOLE ÉLÉMENTAIRE",
                                    "ПОНЕДЕЛЬНИК",
                                    "MIT FREUNDLICHEN GRÜßEN",
                                    "中文"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"besançon",
                                     "école élémentaire",
                                     "понедельник",
                                     "mit freundlichen grüßen",
                                     "中文"};
  test.AddOutput<std::string>("Y", {5}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerInsensitiveFilterUpperMultilingual) {
  // Case-insensitive filtering + UPPER case change (case_change != compare_caseaction_).
  // Exercises the output_filtered_with_wide fallback (cannot reuse cached lowercase wide forms).
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "UPPER", false, {"besançon", "中文"}, test_locale);
  std::vector<int64_t> dims{5};
  std::vector<std::string> input = {"Besançon",  // matches "besançon"
                                    "École élémentaire",
                                    "Понедельник",
                                    "mit freundlichen grüßen",
                                    "中文"};  // matches "中文" (no case, exact match)
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"ÉCOLE ÉLÉMENTAIRE",
                                     "ПОНЕДЕЛЬНИК",
  // Eszett behavior differs by platform
#ifdef __wasm__
                                     "MIT FREUNDLICHEN GRÜẞEN"
#else
                                     "MIT FREUNDLICHEN GRÜßEN"
#endif
  };
  test.AddOutput<std::string>("Y", {3}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerEmptyStringInInput) {
  // Input contains empty strings — should not crash or produce invalid output.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "UPPER", true, {}, test_locale);
  std::vector<int64_t> dims{3};
  std::vector<std::string> input = {"hello", "", "world"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"HELLO", "", "WORLD"};
  test.AddOutput<std::string>("Y", {3}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerSingleElement) {
  // Single-element input tensor with multi-byte UTF-8.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "LOWER", true, {}, test_locale);
  std::vector<int64_t> dims{1};
  std::vector<std::string> input = {"ÉCOLE"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"école"};
  test.AddOutput<std::string>("Y", {1}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerInsensitiveAllFilteredOutMultilingual) {
  // Case-insensitive: all strings match stopwords → output is [1] with empty string.
  // Uses Cyrillic and Chinese stopwords.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "UPPER", false, {"понедельник", "中文", "grüßen"}, test_locale);
  std::vector<int64_t> dims{3};
  std::vector<std::string> input = {"ПОНЕДЕЛЬНИК", "中文", "Grüßen"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output{""};
  test.AddOutput<std::string>("Y", {1}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerInsensitiveMixedCaseStopwords) {
  // Stopwords given in mixed case with accented characters should still match.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "NONE", false, {"ПОНЕДЕЛЬНИК", "École Élémentaire"}, test_locale);
  std::vector<int64_t> dims{4};
  std::vector<std::string> input = {"понедельник",        // matches "ПОНЕДЕЛЬНИК"
                                    "école élémentaire",  // matches "École Élémentaire"
                                    "Besançon",
                                    "中文"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"Besançon", "中文"};
  test.AddOutput<std::string>("Y", {2}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizer2DInputWithFilteringMultilingual) {
  // 2D shape [1, C] with filtering using multilingual input.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "LOWER", true, {"Понедельник"}, test_locale);
  std::vector<int64_t> dims{1, 4};
  std::vector<std::string> input = {"Понедельник", "BESANÇON", "中文", "ÉCOLE"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"besançon", "中文", "école"};
  test.AddOutput<std::string>("Y", {1, 3}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizer2DInputAllFilteredOut) {
  // 2D shape [1, C] with all filtered → output shape [1, 1] with empty string.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "NONE", true, {"中文", "Понедельник"}, test_locale);
  std::vector<int64_t> dims{1, 2};
  std::vector<std::string> input = {"中文", "Понедельник"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output{""};
  test.AddOutput<std::string>("Y", {1, 1}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerInvalidDimensions3D) {
  // Input with 3 dimensions → should fail.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "NONE", true, {}, test_locale);
  std::vector<int64_t> dims{1, 1, 2};
  std::vector<std::string> input = {"hello", "world"};
  test.AddInput<std::string>("T", dims, input);
  test.AddOutput<std::string>("Y", {1, 1, 2}, input);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Input dimensions are either[C > 0] or [1][C > 0] allowed");
}

TEST(ContribOpTest, StringNormalizerInvalidDimensions2DFirstNotOne) {
  // 2D input with first dim != 1 → should fail.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "NONE", true, {}, test_locale);
  std::vector<int64_t> dims{2, 2};
  std::vector<std::string> input = {"a", "b", "c", "d"};
  test.AddInput<std::string>("T", dims, input);
  test.AddOutput<std::string>("Y", {2, 2}, input);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Input dimensions are either[C > 0] or [1][C > 0] allowed");
}

TEST(ContribOpTest, StringNormalizerGermanEszettLower) {
  // German Eszett (ß) lowercasing: ß should remain ß.
  // This tests the converter and case logic with the problematic German character.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "LOWER", true, {}, test_locale);
  std::vector<int64_t> dims{2};
  std::vector<std::string> input = {"GRÜßEN", "STRAßE"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"grüßen", "straße"};
  test.AddOutput<std::string>("Y", {2}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerGermanEszettUpper) {
  // German Eszett (ß) uppercasing: platform-dependent behavior.
  // On wasm, ß uppercases to ẞ (capital eszett U+1E9E).
  // On other platforms, ß remains ß (no single-char uppercase form recognized).
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "UPPER", true, {}, test_locale);
  std::vector<int64_t> dims{2};
  std::vector<std::string> input = {"grüßen", "straße"};
  test.AddInput<std::string>("T", dims, input);

#ifdef __wasm__
  std::vector<std::string> output = {"GRÜẞEN", "STRAẞE"};
#else
  std::vector<std::string> output = {"GRÜßEN", "STRAßE"};
#endif
  test.AddOutput<std::string>("Y", {2}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerInsensitiveGermanEszettFilter) {
  // Case-insensitive filtering with German Eszett in stopwords.
  // "grüßen" lowercased stays "grüßen", should match stopword "grüßen".
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "NONE", false, {"grüßen"}, test_locale);
  std::vector<int64_t> dims{3};
  std::vector<std::string> input = {"Grüßen", "Straße", "中文"};
  test.AddInput<std::string>("T", dims, input);

  // "Grüßen" lowercased → "grüßen" → matches stopword
  std::vector<std::string> output = {"Straße", "中文"};
  test.AddOutput<std::string>("Y", {2}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerCyrillicCaseChange) {
  // Full Cyrillic case conversion test.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "UPPER", true, {}, test_locale);
  std::vector<int64_t> dims{3};
  std::vector<std::string> input = {"понедельник", "Вторник", "среда"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"ПОНЕДЕЛЬНИК", "ВТОРНИК", "СРЕДА"};
  test.AddOutput<std::string>("Y", {3}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(ContribOpTest, StringNormalizerNoStopwordsNoCaseChange) {
  // No stopwords, NONE case change → pure passthrough (fast path).
  // Tests with multilingual content to ensure passthrough preserves bytes exactly.
  OpTester test("StringNormalizer", opset_ver, domain);
  InitTestAttr(test, "NONE", true, {}, test_locale);
  std::vector<int64_t> dims{4};
  std::vector<std::string> input = {"Besançon", "Понедельник", "中文", "grüßen"};
  test.AddInput<std::string>("T", dims, input);

  std::vector<std::string> output = {"Besançon", "Понедельник", "中文", "grüßen"};
  test.AddOutput<std::string>("Y", {4}, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime
