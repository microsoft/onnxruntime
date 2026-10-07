// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_strings.h"
#include "core/platform/telemetry_redaction.h"

#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace onnxruntime::test {
using namespace telemetry_detail;

TEST(TelemetryStringsTest, CapsByteLengthsAndPreservesTighterLimits) {
  for (size_t size : {size_t{1023}, size_t{1024}, size_t{1025}, size_t{1000000}}) {
    const std::string input(size, 'x');
    EXPECT_EQ(BoundedTelemetryString(input).size(), std::min(size, size_t{1024}));
    EXPECT_EQ(BoundedTelemetryString(input, 256).size(), 256);
    EXPECT_EQ(BoundedTelemetryString(input.c_str()).size(), std::min(size, size_t{1024}));
  }
  EXPECT_TRUE(BoundedTelemetryString(nullptr).empty());
  EXPECT_TRUE(BoundedTelemetryString("value", 0).empty());
  const char non_terminated[] = {'a', 'b', 'c'};
  EXPECT_EQ(BoundedTelemetryString(std::string_view(non_terminated, 2)), "ab");
}

TEST(TelemetryStringsTest, DoesNotSplitUtf8Characters) {
  for (const std::string character : {"\xc2\xa2", "\xe2\x82\xac", "\xf0\x9f\x98\x80"}) {
    const std::string exact = std::string(1024 - character.size(), 'a') + character;
    EXPECT_EQ(BoundedTelemetryString(exact + "z"), exact);
    EXPECT_EQ(BoundedTelemetryString((exact + "z").c_str()), exact);
    for (size_t prefix_size = 1025 - character.size(); prefix_size < 1024; ++prefix_size) {
      const std::string prefix(prefix_size, 'a');
      EXPECT_EQ(BoundedTelemetryString(prefix + character), prefix);
      EXPECT_EQ(BoundedTelemetryString((prefix + character).c_str()), prefix);
    }
  }
}

TEST(TelemetryStringsTest, BoundsWideStringsByUtf8Bytes) {
  const std::wstring exact = std::wstring(1021, L'a') + L"\u20ac";
  EXPECT_EQ(TelemetryWideStringView(exact + L"x"), exact);
  EXPECT_EQ(TelemetryWideStringView(std::wstring(1022, L'a') + L"\u20ac").size(), 1022);
  const std::wstring supplementary = L"\U0001f600";
  const std::wstring pair_exact = std::wstring(1020, L'a') + supplementary;
  EXPECT_EQ(TelemetryWideStringView(pair_exact + L"x"), pair_exact);
  EXPECT_EQ(TelemetryWideStringView(std::wstring(1021, L'a') + supplementary).size(), 1021);
  EXPECT_TRUE(TelemetryWideStringView(static_cast<const wchar_t*>(nullptr)).empty());
}

TEST(TelemetryStringsTest, SanitizesMalformedUtf8BeforeStorageAndTransmission) {
  const std::string malformed = "\x80\xC0\xAF\xED\xA0\x80\xF4\x90\x80\x80";
  EXPECT_EQ(BoundedTelemetryString(malformed), "??????????");
  EXPECT_EQ(BoundedTelemetryString(malformed.c_str()), "??????????");
  const std::string continuations(2000, '\x80');
  EXPECT_EQ(BoundedTelemetryString(continuations), std::string(1024, '?'));
  EXPECT_EQ(BoundedTelemetryString(continuations.c_str()), std::string(1024, '?'));
  EXPECT_EQ(BoundedTelemetryString("\xF0\x9F"), "??");
  std::string output = "prefix ";
  EXPECT_TRUE(AppendTelemetryString(output, malformed));
  EXPECT_EQ(output, "prefix ??????????");
  EXPECT_EQ(JoinTelemetryStrings(std::vector<std::string>{malformed, "valid"}), "??????????,valid");
  TelemetryStrings strings;
  EXPECT_STREQ(strings.Utf8(malformed), "??????????");
}

TEST(TelemetryStringsTest, BoundsJniModifiedUtf8AndKeepsSurrogatePairs) {
  const std::u16string supplementary = u"\U0001f600";
  const std::u16string exact = std::u16string(1018, u'a') + supplementary;
  EXPECT_EQ(TelemetryUnicodeStringView(std::u16string_view(exact), 1024, true), exact);
  const std::u16string over = std::u16string(1019, u'a') + supplementary;
  EXPECT_EQ(TelemetryUnicodeStringView(std::u16string_view(over), 1024, true).size(), 1019);
  EXPECT_EQ(TelemetryUnicodeStringView(std::u16string_view(over), 1024, false).size(), over.size());
  const std::u16string nul = std::u16string(1023, u'a') + u'\0';
  EXPECT_EQ(TelemetryUnicodeStringView(std::u16string_view(nul), 1024, true).size(), 1023);
  const std::u16string euro = std::u16string(1022, u'a') + u'\u20ac';
  EXPECT_EQ(TelemetryUnicodeStringView(std::u16string_view(euro), 1024, true).size(), 1022);
}

TEST(TelemetryStringsTest, BoundsJniUnsignedShortCodeUnitsWithoutCharTraits) {
  const std::array<uint16_t, 4> value{0x61, 0xD83D, 0xDE00, 0};
  EXPECT_EQ(TelemetryUnicodePrefixLength(value.data(), value.size(), 6, true), size_t{1});
  EXPECT_EQ(TelemetryUnicodePrefixLength(value.data(), value.size(), 7, true), size_t{3});
  EXPECT_EQ(TelemetryUnicodePrefixLength(value.data(), value.size(), 8, true), size_t{3});
  EXPECT_EQ(TelemetryUnicodePrefixLength(value.data(), value.size(), 9, true), value.size());
}

TEST(TelemetryStringsTest, AppendDoesNotExceedBudgetOrSplitCharacters) {
  std::string output(1023, 'a');
  EXPECT_FALSE(AppendTelemetryString(output, "\xc2\xa2"));
  EXPECT_EQ(output.size(), 1023);
  EXPECT_TRUE(AppendTelemetryString(output, "b"));
  EXPECT_FALSE(AppendTelemetryString(output, "c"));
  EXPECT_EQ(output.size(), 1024);
  EXPECT_TRUE(AppendTelemetryString(output, ""));
  output.assign(100000, 'a');
  EXPECT_FALSE(AppendTelemetryString(output, "b"));
  EXPECT_EQ(output.size(), 1024);
}

TEST(TelemetryStringsTest, BoundsAggregatesAndCollectionCounts) {
  const std::vector<std::string> empty_values(100000);
  EXPECT_EQ(JoinTelemetryStrings(empty_values), std::string(127, ','));
  EXPECT_EQ(JoinTelemetryStrings(std::vector<std::string>{"a", "b"}), "a,b");
  EXPECT_EQ(JoinTelemetryStrings(std::vector<std::string>{std::string(100000, 'x'), "b"}).size(), 1024);
  const std::map<std::string, std::string> values{{"b", "two"}, {"a", "one"}};
  EXPECT_EQ(FormatTelemetryMap(values), "a=one,b=two");
  EXPECT_EQ(FormatTelemetryMap(values, ",", ":"), "a:one,b:two");
  const std::map<std::string, std::string> large{{std::string(100000, 'x'), std::string(100000, 'y')}};
  EXPECT_EQ(FormatTelemetryMap(large).size(), 1024);
  std::map<std::string, std::string> many;
  for (int i = 0; i < 1000; ++i) many.emplace(std::to_string(i), "");
  bool truncated = false;
  const auto formatted = FormatTelemetryMap(many, ",", "=", &truncated);
  EXPECT_TRUE(truncated);
  EXPECT_EQ(std::count(formatted.begin(), formatted.end(), '='), 128);
}

TEST(TelemetryStringsTest, AppendsCompleteAlignedRows) {
  std::array<std::string, 3> summaries;
  const std::array outputs{&summaries[0], &summaries[1], &summaries[2]};
  EXPECT_TRUE(AppendTelemetryRow(outputs, std::array<std::string_view, 3>{"CPU", "0x0000", "CPUExecutionProvider:"},
                                 true));
  EXPECT_TRUE(AppendTelemetryRow(outputs, std::array<std::string_view, 3>{"GPU", "0x10DE", "CUDAExecutionProvider:1"},
                                 false));
  EXPECT_EQ(summaries[0], "CPU,GPU");
  EXPECT_EQ(summaries[1], "0x0000,0x10DE");
  EXPECT_EQ(summaries[2], "CPUExecutionProvider:,CUDAExecutionProvider:1");
}

TEST(TelemetryStringsTest, RejectsPartialRowsWithoutChangingAnySummary) {
  const std::array<std::string_view, 3> row{"CPU", "0x0000", "CPUExecutionProvider:"};
  for (size_t full_column = 0; full_column < row.size(); ++full_column) {
    for (size_t size : {kMaxTelemetryStringLength, kMaxTelemetryStringLength - 1,
                        kMaxTelemetryStringLength - row[full_column].size()}) {
      SCOPED_TRACE(full_column);
      SCOPED_TRACE(size);
      std::array<std::string, 3> summaries{"type", "vendor", "version"};
      summaries[full_column].assign(size, 'x');
      const auto before = summaries;
      EXPECT_FALSE(AppendTelemetryRow(std::array{&summaries[0], &summaries[1], &summaries[2]}, row, false));
      EXPECT_EQ(summaries, before);
    }
  }
}

TEST(TelemetryStringsTest, RejectsOversizedFirstRowsAndIncompleteUtf8Rows) {
  for (size_t column = 0; column < 3; ++column) {
    SCOPED_TRACE(column);
    std::array<std::string, 3> summaries;
    std::array<std::string, 3> row{"CPU", "0x0000", "CPUExecutionProvider:"};
    row[column].assign(kMaxTelemetryStringLength + 1, 'x');
    const std::array<std::string_view, 3> values{row[0], row[1], row[2]};
    EXPECT_FALSE(AppendTelemetryRow(std::array{&summaries[0], &summaries[1], &summaries[2]}, values, true));
    EXPECT_EQ(summaries, (std::array<std::string, 3>{}));

    summaries[column].assign(kMaxTelemetryStringLength - 2, 'x');
    row[column] = "\xe2\x82\xac";
    const auto before = summaries;
    EXPECT_FALSE(AppendTelemetryRow(std::array{&summaries[0], &summaries[1], &summaries[2]},
                                    std::array<std::string_view, 3>{row[0], row[1], row[2]}, false));
    EXPECT_EQ(summaries, before);
  }
}

TEST(TelemetryStringsTest, AcceptsExactBudgetRowsAndNormalizesMalformedUtf8) {
  std::array<std::string, 3> summaries;
  const std::array outputs{&summaries[0], &summaries[1], &summaries[2]};
  EXPECT_TRUE(AppendTelemetryRow(outputs, std::array<std::string_view, 3>{"", "\x80", "version"}, true));
  EXPECT_EQ(summaries, (std::array<std::string, 3>{"", "?", "version"}));
  const std::array<std::string, 3> row{
      std::string(kMaxTelemetryStringLength - 1, 't'),
      std::string(kMaxTelemetryStringLength - 2, 'v'),
      std::string(kMaxTelemetryStringLength - 8, 'r')};
  EXPECT_TRUE(AppendTelemetryRow(outputs, std::array<std::string_view, 3>{row[0], row[1], row[2]}, false));
  for (const auto& summary : summaries) {
    EXPECT_EQ(summary.size(), kMaxTelemetryStringLength);
    EXPECT_EQ(std::count(summary.begin(), summary.end(), ','), 1);
  }
}

TEST(TelemetryStringsTest, RetainsPointersAcrossAdditionalFields) {
  TelemetryStrings strings;
  const char* first = strings.Utf8("first");
  const wchar_t* wide_first = strings.Wide(L"first");
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(std::string_view(strings.Utf8(std::string(10000, 'a'))).size(), 1024);
    strings.Wide(std::wstring(10000, L'a'));
  }
  EXPECT_STREQ(first, "first");
  EXPECT_STREQ(wide_first, L"first");
}

TEST(TelemetryStringsTest, PropagatesMapTruncationToRedaction) {
  const std::map<std::string, std::string> options{{"cache", "alice/" + std::string(2000, 'a') + "/model"}};
  bool truncated = false;
  const auto formatted = FormatTelemetryMap(options, ",", ":", &truncated);
  EXPECT_TRUE(truncated);
  EXPECT_EQ(ScrubStringForTelemetry(formatted, truncated), "[path]");
  FormatTelemetryMap(std::map<std::string, std::string>{{"a", "b"}}, ",", ":", &truncated);
  EXPECT_FALSE(truncated);
}
}  // namespace onnxruntime::test
