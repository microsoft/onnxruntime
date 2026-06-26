// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_redaction.h"

#include <string>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(TelemetryRedactionTest, EmptyAndNoPath) {
  EXPECT_EQ(ScrubErrorMessage(""), "");
  EXPECT_EQ(ScrubErrorMessage("no path here"), "no path here");
  EXPECT_EQ(ScrubErrorMessage("error code 13"), "error code 13");
}

TEST(TelemetryRedactionTest, PosixPathReplacedWithPlaceholder) {
  EXPECT_EQ(ScrubErrorMessage("Load model from /home/alice/models/foo.onnx failed"),
            "Load model from [path] failed");
  // The username must not survive.
  EXPECT_EQ(ScrubErrorMessage("/home/alice/models/foo.onnx").find("alice"), std::string::npos);
}

TEST(TelemetryRedactionTest, WindowsDriveAndUncReplaced) {
  EXPECT_EQ(ScrubErrorMessage("Load C:\\Users\\bob\\m.onnx failed"), "Load [path] failed");
  EXPECT_EQ(ScrubErrorMessage("open D:/data/secret/model.onnx"), "open [path]");
  EXPECT_EQ(ScrubErrorMessage("from \\\\server\\share\\dir\\weights.bin done"), "from [path] done");
  EXPECT_EQ(ScrubErrorMessage("Load C:\\Users\\bob\\m.onnx failed").find("bob"), std::string::npos);
}

TEST(TelemetryRedactionTest, PathsWithSpacesDoNotLeakUsername) {
  // Both halves of a spaced path contain a backslash, so each is replaced; no username leaks.
  EXPECT_EQ(ScrubErrorMessage("Load C:\\Users\\First Last\\model.onnx failed"),
            "Load [path] [path] failed");
  EXPECT_EQ(ScrubErrorMessage("Load C:\\Users\\First Last\\model.onnx failed").find("First"),
            std::string::npos);
}

TEST(TelemetryRedactionTest, MultiSegmentRelativeAndUrlReplaced) {
  // Matches onnxruntime-genai: a token with 2+ "/x" segments (incl. URLs) is treated as a path.
  EXPECT_EQ(ScrubErrorMessage("a/b/c"), "[path]");
  EXPECT_EQ(ScrubErrorMessage("see https://example.com/a/b/c for details"), "see [path] for details");
  EXPECT_EQ(ScrubErrorMessage("input:/home/alice/secret/m.onnx"), "[path]");
  EXPECT_EQ(ScrubErrorMessage("file:///home/alice/secret/model.onnx"), "[path]");
  EXPECT_EQ(ScrubErrorMessage("~/.config/app/x"), "[path]");
}

TEST(TelemetryRedactionTest, SingleSegmentAndNonPathSlashesKept) {
  EXPECT_EQ(ScrubErrorMessage("models/foo.onnx"), "models/foo.onnx");
  EXPECT_EQ(ScrubErrorMessage("ratio 3/4 and and/or"), "ratio 3/4 and and/or");
}

TEST(TelemetryRedactionTest, LengthIsCappedAfterScrub) {
  const std::string long_msg(300, 'x');
  EXPECT_EQ(ScrubErrorMessage(long_msg).size(), kMaxTelemetryErrorMessageLength);
  EXPECT_LE(ScrubErrorMessage("short").size(), kMaxTelemetryErrorMessageLength);
}

}  // namespace test
}  // namespace onnxruntime
