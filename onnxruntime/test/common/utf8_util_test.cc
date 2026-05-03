// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/utf8_util.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

struct Sample {
  const char* sequence;
  bool valid;
};

const std::vector<Sample> samples = {
    {"a", true},
    {"\xc3\xb1", true},
    {"\xc3\x28", false},
    {"\xa0\xa1", false},
    {"\xe2\x82\xa1", true},
    {"\xe2\x28\xa1", false},
    {"\xe2\x82\x28", false},
    {"\xf0\x90\x8c\xbc", true},
    {"\xf0\x28\x8c\xbc", false},
    {"\xf0\x90\x28\xbc", false},
    {"\xf0\x28\x8c\x28", false},
    {"\xf8\xa1\xa1\xa1\xa1", false},       // valid but not Unicode
    {"\xfc\xa1\xa1\xa1\xa1\xa1", false}};  // valid but not Unicode

TEST(Utf8UtilTest, Validate) {
  using namespace utf8_util;
  for (auto& s : samples) {
    size_t utf8_len = 0;
    if (s.valid != utf8_validate(reinterpret_cast<const unsigned char*>(s.sequence), strnlen(s.sequence, onnxruntime::kMaxStrLen), utf8_len)) {
      ASSERT_TRUE(false);
    } else {
      if (s.valid) {
        ASSERT_EQ(1U, utf8_len);
      }
    }
  }
}

// --- utf8_bytes tests ---

TEST(Utf8UtilTest, Utf8Bytes_Ascii) {
  using namespace utf8_util;
  size_t len = 0;
  // All ASCII bytes (0x00-0x7F) should be 1-byte
  EXPECT_TRUE(utf8_bytes(0x00, len));
  EXPECT_EQ(1U, len);
  EXPECT_TRUE(utf8_bytes('A', len));
  EXPECT_EQ(1U, len);
  EXPECT_TRUE(utf8_bytes(0x7F, len));
  EXPECT_EQ(1U, len);
}

TEST(Utf8UtilTest, Utf8Bytes_TwoByte) {
  using namespace utf8_util;
  size_t len = 0;
  // 0xC0-0xDF are 2-byte lead bytes
  EXPECT_TRUE(utf8_bytes(0xC0, len));
  EXPECT_EQ(2U, len);
  EXPECT_TRUE(utf8_bytes(0xC2, len));
  EXPECT_EQ(2U, len);
  EXPECT_TRUE(utf8_bytes(0xDF, len));
  EXPECT_EQ(2U, len);
}

TEST(Utf8UtilTest, Utf8Bytes_ThreeByte) {
  using namespace utf8_util;
  size_t len = 0;
  // 0xE0-0xEF are 3-byte lead bytes
  EXPECT_TRUE(utf8_bytes(0xE0, len));
  EXPECT_EQ(3U, len);
  EXPECT_TRUE(utf8_bytes(0xED, len));
  EXPECT_EQ(3U, len);
  EXPECT_TRUE(utf8_bytes(0xEF, len));
  EXPECT_EQ(3U, len);
}

TEST(Utf8UtilTest, Utf8Bytes_FourByte) {
  using namespace utf8_util;
  size_t len = 0;
  // 0xF0-0xF7 are 4-byte lead bytes
  EXPECT_TRUE(utf8_bytes(0xF0, len));
  EXPECT_EQ(4U, len);
  EXPECT_TRUE(utf8_bytes(0xF4, len));
  EXPECT_EQ(4U, len);
  EXPECT_TRUE(utf8_bytes(0xF7, len));
  EXPECT_EQ(4U, len);
}

TEST(Utf8UtilTest, Utf8Bytes_Invalid) {
  using namespace utf8_util;
  size_t len = 99;
  // Continuation bytes (0x80-0xBF) are not valid lead bytes
  EXPECT_FALSE(utf8_bytes(0x80, len));
  EXPECT_FALSE(utf8_bytes(0xBF, len));
  // 0xF8-0xFF are invalid (would be 5+ byte sequences)
  EXPECT_FALSE(utf8_bytes(0xF8, len));
  EXPECT_FALSE(utf8_bytes(0xF9, len));
  EXPECT_FALSE(utf8_bytes(0xFC, len));
  EXPECT_FALSE(utf8_bytes(0xFE, len));
  EXPECT_FALSE(utf8_bytes(0xFF, len));
}

// --- utf8_len tests ---

TEST(Utf8UtilTest, Utf8Len_Empty) {
  using namespace utf8_util;
  size_t len = 99;
  EXPECT_TRUE(utf8_len(reinterpret_cast<const unsigned char*>(""), 0, len));
  EXPECT_EQ(0U, len);
}

TEST(Utf8UtilTest, Utf8Len_Ascii) {
  using namespace utf8_util;
  size_t len = 0;
  const char* s = "Hello";
  EXPECT_TRUE(utf8_len(reinterpret_cast<const unsigned char*>(s), 5, len));
  EXPECT_EQ(5U, len);
}

TEST(Utf8UtilTest, Utf8Len_Multibyte) {
  using namespace utf8_util;
  size_t len = 0;
  // "café" = 'c' 'a' 'f' U+00E9(2 bytes) = 5 bytes, 4 chars
  const char* s = "caf\xc3\xa9";
  EXPECT_TRUE(utf8_len(reinterpret_cast<const unsigned char*>(s), 5, len));
  EXPECT_EQ(4U, len);
}

TEST(Utf8UtilTest, Utf8Len_ThreeByteChars) {
  using namespace utf8_util;
  size_t len = 0;
  // U+4E16 (世) = 0xE4 0xB8 0x96, U+754C (界) = 0xE7 0x95 0x8C
  const char* s = "\xe4\xb8\x96\xe7\x95\x8c";  // "世界"
  EXPECT_TRUE(utf8_len(reinterpret_cast<const unsigned char*>(s), 6, len));
  EXPECT_EQ(2U, len);
}

TEST(Utf8UtilTest, Utf8Len_FourByteChars) {
  using namespace utf8_util;
  size_t len = 0;
  // U+1F600 (😀) = 0xF0 0x9F 0x98 0x80
  const char* s = "\xf0\x9f\x98\x80";
  EXPECT_TRUE(utf8_len(reinterpret_cast<const unsigned char*>(s), 4, len));
  EXPECT_EQ(1U, len);
}

TEST(Utf8UtilTest, Utf8Len_Mixed) {
  using namespace utf8_util;
  size_t len = 0;
  // "A" (1) + U+00F1 (2) + U+4E16 (3) + U+1F600 (4) = 10 bytes, 4 chars
  const char* s = "A\xc3\xb1\xe4\xb8\x96\xf0\x9f\x98\x80";
  EXPECT_TRUE(utf8_len(reinterpret_cast<const unsigned char*>(s), 10, len));
  EXPECT_EQ(4U, len);
}

TEST(Utf8UtilTest, Utf8Len_InvalidLeadByte) {
  using namespace utf8_util;
  size_t len = 0;
  // 0xF8 is invalid lead byte
  const char* s = "\xf8\x80\x80\x80";
  EXPECT_FALSE(utf8_len(reinterpret_cast<const unsigned char*>(s), 4, len));
}

TEST(Utf8UtilTest, Utf8Len_Truncated) {
  using namespace utf8_util;
  size_t len = 0;
  // 2-byte sequence but only 1 byte available
  const char* s = "\xc3";
  EXPECT_FALSE(utf8_len(reinterpret_cast<const unsigned char*>(s), 1, len));
}

// --- utf8_validate additional tests ---

TEST(Utf8UtilTest, Validate_EmptyString) {
  using namespace utf8_util;
  size_t chars = 99;
  EXPECT_TRUE(utf8_validate(reinterpret_cast<const unsigned char*>(""), 0, chars));
  EXPECT_EQ(0U, chars);
}

TEST(Utf8UtilTest, Validate_MultiCharString) {
  using namespace utf8_util;
  size_t chars = 0;
  // "Héllo" = 'H' U+00E9(2b) 'l' 'l' 'o' = 6 bytes, 5 chars
  const char* s = "H\xc3\xa9llo";
  EXPECT_TRUE(utf8_validate(reinterpret_cast<const unsigned char*>(s), 6, chars));
  EXPECT_EQ(5U, chars);
}

TEST(Utf8UtilTest, Validate_OverlongTwoByte) {
  using namespace utf8_util;
  size_t chars = 0;
  // Overlong encoding of U+0000: 0xC0 0x80 (should be rejected)
  const char* s = "\xc0\x80";
  EXPECT_FALSE(utf8_validate(reinterpret_cast<const unsigned char*>(s), 2, chars));
}

TEST(Utf8UtilTest, Validate_SurrogatePair) {
  using namespace utf8_util;
  size_t chars = 0;
  // U+D800 encoded as 3-byte: 0xED 0xA0 0x80 (invalid surrogate)
  const char* s = "\xed\xa0\x80";
  EXPECT_FALSE(utf8_validate(reinterpret_cast<const unsigned char*>(s), 3, chars));
}

TEST(Utf8UtilTest, Validate_MaxCodepoint) {
  using namespace utf8_util;
  size_t chars = 0;
  // U+10FFFF = 0xF4 0x8F 0xBF 0xBF (valid, max Unicode codepoint)
  const char* s = "\xf4\x8f\xbf\xbf";
  EXPECT_TRUE(utf8_validate(reinterpret_cast<const unsigned char*>(s), 4, chars));
  EXPECT_EQ(1U, chars);
}

TEST(Utf8UtilTest, Validate_BeyondMaxCodepoint) {
  using namespace utf8_util;
  size_t chars = 0;
  // U+110000 = 0xF4 0x90 0x80 0x80 (invalid, beyond U+10FFFF)
  const char* s = "\xf4\x90\x80\x80";
  EXPECT_FALSE(utf8_validate(reinterpret_cast<const unsigned char*>(s), 4, chars));
}

TEST(Utf8UtilTest, Validate_ContinuationByteAlone) {
  using namespace utf8_util;
  size_t chars = 0;
  // A lone continuation byte
  const char* s = "\x80";
  EXPECT_FALSE(utf8_validate(reinterpret_cast<const unsigned char*>(s), 1, chars));
}

// --- Non-Windows conversion tests ---
#ifndef _WIN32

TEST(Utf8UtilTest, WideToUtf8RequiredSize_Ascii) {
  std::wstring ws = L"Hello";
  EXPECT_EQ(5U, WideToUtf8RequiredSize(ws));
}

TEST(Utf8UtilTest, WideToUtf8RequiredSize_Multibyte) {
  // U+00E9 -> 2 bytes, U+4E16 -> 3 bytes, U+1F600 -> 4 bytes
  std::wstring ws;
  ws += static_cast<wchar_t>(0x00E9);   // 2 bytes
  ws += static_cast<wchar_t>(0x4E16);   // 3 bytes
  ws += static_cast<wchar_t>(0x1F600);  // 4 bytes
  EXPECT_EQ(9U, WideToUtf8RequiredSize(ws));
}

TEST(Utf8UtilTest, WideToUtf8_RoundTrip_Ascii) {
  std::wstring ws = L"Hello World";
  std::string result;
  result.resize(WideToUtf8RequiredSize(ws));
  ASSERT_TRUE(WideToUtf8(ws, result).IsOK());
  EXPECT_EQ("Hello World", result);
}

TEST(Utf8UtilTest, WideToUtf8_RoundTrip_Multibyte) {
  // Build wide string with various codepoints
  std::wstring ws;
  ws += static_cast<wchar_t>(0x00E9);   // é
  ws += static_cast<wchar_t>(0x4E16);   // 世
  ws += static_cast<wchar_t>(0x1F600);  // 😀

  std::string utf8;
  utf8.resize(WideToUtf8RequiredSize(ws));
  ASSERT_TRUE(WideToUtf8(ws, utf8).IsOK());

  // Verify via round-trip
  std::wstring back;
  back.resize(utf8.size());
  ASSERT_TRUE(Utf8ToWide(utf8, back).IsOK());
  EXPECT_EQ(ws, back);
}

TEST(Utf8UtilTest, WideToUtf8_Empty) {
  std::wstring ws;
  std::string result = "notempty";
  ASSERT_TRUE(WideToUtf8(ws, result).IsOK());
  EXPECT_TRUE(result.empty());
}

TEST(Utf8UtilTest, Utf8ToWide_Empty) {
  std::string s;
  std::wstring result = L"notempty";
  ASSERT_TRUE(Utf8ToWide(s, result).IsOK());
  EXPECT_TRUE(result.empty());
}

TEST(Utf8UtilTest, Utf8ToWide_Ascii) {
  std::string s = "ABC";
  std::wstring result;
  result.resize(s.size());
  ASSERT_TRUE(Utf8ToWide(s, result).IsOK());
  EXPECT_EQ(L"ABC", result);
}

TEST(Utf8UtilTest, Utf8ToWide_TruncatedSequence) {
  // 3-byte sequence missing last byte
  std::string s = "\xe4\xb8";
  std::wstring result;
  result.resize(s.size());
  EXPECT_FALSE(Utf8ToWide(s, result).IsOK());
}

TEST(Utf8UtilTest, Utf8ToWide_InvalidContinuationByte) {
  // 2-byte lead 0xC3 followed by non-continuation 0x28
  std::string s = "\xc3\x28";
  std::wstring result;
  result.resize(s.size());
  EXPECT_FALSE(Utf8ToWide(s, result).IsOK());
}

TEST(Utf8UtilTest, Utf8ToWide_OverlongEncoding) {
  // Overlong 2-byte for U+002F ('/') = 0xC0 0xAF
  std::string s = "\xc0\xaf";
  std::wstring result;
  result.resize(s.size());
  EXPECT_FALSE(Utf8ToWide(s, result).IsOK());
}

TEST(Utf8UtilTest, Utf8ToWide_SurrogateCodepoint) {
  // U+D800 as 3-byte UTF-8: 0xED 0xA0 0x80
  std::string s = "\xed\xa0\x80";
  std::wstring result;
  result.resize(s.size());
  EXPECT_FALSE(Utf8ToWide(s, result).IsOK());
}

TEST(Utf8UtilTest, Utf8ToWide_BeyondUnicode) {
  // U+110000: 0xF4 0x90 0x80 0x80
  std::string s = "\xf4\x90\x80\x80";
  std::wstring result;
  result.resize(s.size());
  EXPECT_FALSE(Utf8ToWide(s, result).IsOK());
}

TEST(Utf8UtilTest, Utf8ToWide_InvalidLeadByte) {
  // 0xF8 is not a valid UTF-8 lead byte
  std::string s = "\xf8\x80\x80\x80\x80";
  std::wstring result;
  result.resize(s.size());
  EXPECT_FALSE(Utf8ToWide(s, result).IsOK());
}

TEST(Utf8UtilTest, Utf8ToWideString_ValidInput) {
  std::string s = "caf\xc3\xa9";  // "café"
  std::wstring result = Utf8ToWideString(s);
  EXPECT_EQ(4U, result.size());
  EXPECT_EQ(static_cast<wchar_t>('c'), result[0]);
  EXPECT_EQ(static_cast<wchar_t>('a'), result[1]);
  EXPECT_EQ(static_cast<wchar_t>('f'), result[2]);
  EXPECT_EQ(static_cast<wchar_t>(0x00E9), result[3]);
}

TEST(Utf8UtilTest, Utf8ToWideString_InvalidInput) {
  // Should throw on invalid UTF-8
  std::string s = "\xc0\xaf";
  EXPECT_THROW(Utf8ToWideString(s), OnnxRuntimeException);
}

#endif  // !_WIN32

}  // namespace test
}  // namespace onnxruntime
