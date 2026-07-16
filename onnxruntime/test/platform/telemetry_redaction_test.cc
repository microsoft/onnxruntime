// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_redaction.h"

#include <string>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// The scrubber is anchor-based: it finds the FIRST filesystem-path anchor (drive prefix, UNC prefix,
// home prefix, a relative Windows path with >=2 '\'-delimited segments, or a >=2-segment POSIX path) and
// replaces everything from that anchor to the end of the message with a single "[path]" placeholder.
// Redacting to end-of-message -- rather than classifying each whitespace-delimited token -- is what makes
// a space-separated user name (C:\Users\First Last\...) impossible to leak.

TEST(TelemetryRedactionTest, EmptyAndNoPath) {
  EXPECT_EQ(ScrubStringForTelemetry(""), "");
  EXPECT_EQ(ScrubStringForTelemetry("no path here"), "no path here");
  EXPECT_EQ(ScrubStringForTelemetry("error code 13"), "error code 13");
}

TEST(TelemetryRedactionTest, AbsolutePosixAndHomePathsFullyRedacted) {
  // An anchor at the very start collapses the whole string to "[path]"; nothing (not even the parent
  // directory or file name) is retained, so a user name can never survive.
  EXPECT_EQ(ScrubStringForTelemetry("/home/alice/model.onnx"), "[path]");
  EXPECT_EQ(ScrubStringForTelemetry("/home/alice/proj/rn/model.onnx"), "[path]");
  EXPECT_EQ(ScrubStringForTelemetry("/data/models/rn/model.onnx"), "[path]");
  EXPECT_EQ(ScrubStringForTelemetry("/data/models/secret/"), "[path]");
  EXPECT_EQ(ScrubStringForTelemetry("~/.config/app/x"), "[path]");
}

TEST(TelemetryRedactionTest, WindowsDriveAndUncFullyRedacted) {
  EXPECT_EQ(ScrubStringForTelemetry("C:\\Users\\bob\\model.onnx"), "[path]");
  EXPECT_EQ(ScrubStringForTelemetry("C:/Users\\bob\\model.onnx"), "[path]");
  EXPECT_EQ(ScrubStringForTelemetry("\\\\server\\share\\dir\\weights.bin"), "[path]");
}

TEST(TelemetryRedactionTest, EmbeddedPathRedactedToEndOfMessage) {
  // Everything before the anchor is kept verbatim; everything from the anchor onward (including any
  // trailing prose after the path) is replaced, because a path may contain spaces.
  EXPECT_EQ(ScrubStringForTelemetry("Load model from /home/alice/models/foo.onnx failed"),
            "Load model from [path]");
  EXPECT_EQ(ScrubStringForTelemetry("Load C:\\proj\\bin\\m.onnx failed"), "Load [path]");
  EXPECT_EQ(ScrubStringForTelemetry("open D:/data/secret/model.onnx"), "open [path]");
  EXPECT_EQ(ScrubStringForTelemetry("from \\\\server\\share\\dir\\weights.bin done"), "from [path]");
}

TEST(TelemetryRedactionTest, SpacedUsernameFullyRedacted) {
  // A space inside a path used to split it into separate tokens and let the trailing name half
  // ("Last") leak. Anchoring on the drive prefix and redacting to end-of-message removes both halves.
  const std::string spaced = ScrubStringForTelemetry("Load C:\\Users\\First Last\\model.onnx failed");
  EXPECT_EQ(spaced, "Load [path]");
  EXPECT_EQ(spaced.find("First"), std::string::npos);
  EXPECT_EQ(spaced.find("Last"), std::string::npos);
}

TEST(TelemetryRedactionTest, HomeUsernameNeverLeaksAcrossVariants) {
  // Case, mixed/duplicate separators, "." segments, and embedding in a larger token must never leave
  // the user name in the output. Exact output varies with where the first anchor lands, so the
  // security-relevant invariant is asserted directly: the user name is absent and a redaction occurred.
  const char* inputs[] = {
      "C:\\UsErS\\alice\\model.onnx",
      "C:\\Users/alice\\model.onnx",
      "/UsErS/alice/model.onnx",
      "C:\\Users/alice/proj\\model.onnx",
      "/home//alice/model.onnx",
      "C:\\Users\\\\alice\\model.onnx",
      "/home/./alice/model.onnx",
      "input:/home/alice/secret/m.onnx",
      "file:///home/alice/secret/model.onnx",
      "Users\\alice\\model.onnx",
      "at proj\\alice\\weights\\m.onnx",
  };
  for (const char* in : inputs) {
    const std::string out = ScrubStringForTelemetry(in);
    EXPECT_EQ(out.find("alice"), std::string::npos) << "user name leaked for input: " << in;
    EXPECT_NE(out.find("[path]"), std::string::npos) << "no redaction happened for input: " << in;
  }
}

TEST(TelemetryRedactionTest, MultiSegmentRelativePathReplaced) {
  // A relative token with 2+ "/x" segments is a path anchor; the character(s) before the leading
  // slash are kept and the rest is redacted.
  EXPECT_EQ(ScrubStringForTelemetry("a/b/c"), "a[path]");
  EXPECT_EQ(ScrubStringForTelemetry("x/y/z/"), "x[path]");
}

TEST(TelemetryRedactionTest, RelativeWindowsPathReplaced) {
  // A drive-less relative Windows path (>= 2 '\'-delimited segments) has no C:\ / UNC / home prefix to
  // anchor on, but is still a path: it anchors at the first backslash and redacts to end-of-message, so
  // the user name in the second segment cannot leak.
  EXPECT_EQ(ScrubStringForTelemetry("a\\b\\c"), "a[path]");
  EXPECT_EQ(ScrubStringForTelemetry("Users\\alice\\model.onnx"), "Users[path]");
  EXPECT_EQ(ScrubStringForTelemetry("Load Users\\bob\\m.onnx failed"), "Load Users[path]");
}

TEST(TelemetryRedactionTest, SingleSegmentAndNonPathSlashesKept) {
  // A single "/x" segment is not enough to anchor a path, so ordinary text with slashes is preserved.
  EXPECT_EQ(ScrubStringForTelemetry("models/foo.onnx"), "models/foo.onnx");
  EXPECT_EQ(ScrubStringForTelemetry("ratio 3/4 and and/or"), "ratio 3/4 and and/or");
  // A single backslash is likewise not a path anchor, so a Windows account name (DOMAIN\user) and other
  // one-backslash tokens are kept verbatim rather than over-redacted.
  EXPECT_EQ(ScrubStringForTelemetry("domain\\user"), "domain\\user");
  EXPECT_EQ(ScrubStringForTelemetry("read\\write access"), "read\\write access");
}

TEST(TelemetryRedactionTest, LengthIsCappedAfterScrub) {
  const std::string long_msg(300, 'x');
  EXPECT_EQ(ScrubStringForTelemetry(long_msg).size(), kMaxTelemetryStringLength);
  EXPECT_LE(ScrubStringForTelemetry("short").size(), kMaxTelemetryStringLength);
}

}  // namespace test
}  // namespace onnxruntime
