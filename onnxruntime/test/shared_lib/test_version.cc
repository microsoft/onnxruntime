// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_config.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_version_check.h"

#include "gtest/gtest.h"

using onnxruntime::version_check::IsOrtVersionValid;
using onnxruntime::version_check::ParseUint;

// Compile-time tests for ParseUint
static_assert(ParseUint("0") == 0u);
static_assert(ParseUint("1") == 1u);
static_assert(ParseUint("25") == 25u);
static_assert(ParseUint("123") == 123u);
static_assert(ParseUint("4294967295") == 4294967295u);  // UINT32_MAX
static_assert(!(ParseUint("4294967296").has_value));    // UINT32_MAX + 1 overflows
static_assert(!(ParseUint("").has_value));              // empty
static_assert(!(ParseUint("01").has_value));            // leading zero
static_assert(!(ParseUint("00").has_value));            // leading zero
static_assert(!(ParseUint("abc").has_value));           // non-digit
static_assert(!(ParseUint("1a").has_value));            // trailing non-digit
static_assert(!(ParseUint("-1").has_value));            // negative sign
static_assert(!(ParseUint("1.0").has_value));           // contains dot
static_assert(ParseUint("0").has_value);
static_assert(!ParseUint("").has_value);

// Compile-time tests for IsOrtVersionValid (default expected_api_version = ORT_API_VERSION)
static_assert(IsOrtVersionValid(ORT_VERSION));  // current version must be valid

// Invalid formats
static_assert(!IsOrtVersionValid(""));
static_assert(!IsOrtVersionValid("1"));
static_assert(!IsOrtVersionValid("1.0"));
static_assert(!IsOrtVersionValid("1.0.0.0"));  // too many dots
static_assert(!IsOrtVersionValid("2.0.0"));    // major != 1
static_assert(!IsOrtVersionValid("1.02.0"));   // leading zero in minor
static_assert(!IsOrtVersionValid("1.0.01"));   // leading zero in patch
static_assert(!IsOrtVersionValid("1..0"));     // empty minor
static_assert(!IsOrtVersionValid("1.0."));     // empty patch
static_assert(!IsOrtVersionValid(".1.0"));     // empty major
static_assert(!IsOrtVersionValid("abc"));      // non-numeric
static_assert(!IsOrtVersionValid("1.abc.0"));  // non-numeric minor
static_assert(!IsOrtVersionValid("1.0.abc"));  // non-numeric patch

// Compile-time tests for IsOrtVersionValid with explicit expected_api_version
static_assert(IsOrtVersionValid("1.0.0", 0));
static_assert(IsOrtVersionValid("1.1.0", 1));
static_assert(IsOrtVersionValid("1.25.0", 25));
static_assert(IsOrtVersionValid("1.25.3", 25));
static_assert(IsOrtVersionValid("1.100.0", 100));
static_assert(!IsOrtVersionValid("1.25.0", 24));  // minor doesn't match expected
static_assert(!IsOrtVersionValid("1.25.0", 26));  // minor doesn't match expected
static_assert(!IsOrtVersionValid("1.0.0", 1));    // minor 0 != expected 1
static_assert(!IsOrtVersionValid("2.0.0", 0));    // major != 1
static_assert(!IsOrtVersionValid("1.02.0", 2));   // leading zero in minor
static_assert(!IsOrtVersionValid("1.0.01", 0));   // leading zero in patch

TEST(CApiTest, VersionIsValid) {
  // Runtime sanity check — the version string returned by the API is the expected one.
  EXPECT_STREQ(Ort::GetVersionString().c_str(), ORT_VERSION);
}
