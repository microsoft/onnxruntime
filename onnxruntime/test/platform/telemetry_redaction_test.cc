// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_redaction.h"

#include <string>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(TelemetryRedactionTest, EmptyAndNoPath) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry(""), "");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("no path here"), "no path here");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("error code 13"), "error code 13");
}

TEST(TelemetryRedactionTest, PosixAbsolutePathReducedToBasename) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("Load model from /home/alice/models/foo.onnx failed"),
            "Load model from foo.onnx failed");
  // The username in the directory is dropped.
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/var/lib/onnxruntime/cache/x.bin").find("lib"),
            std::string::npos);
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/tmp/onnxruntime_telemetry_cache/db"), "db");
}

TEST(TelemetryRedactionTest, WindowsDrivePathReducedToBasename) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("Load C:\\Users\\bob\\m.onnx failed"),
            "Load m.onnx failed");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("open D:/data/secret/model.onnx"),
            "open model.onnx");
  // Username 'bob' must not survive.
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("C:\\Users\\bob\\m.onnx").find("bob"),
            std::string::npos);
}

TEST(TelemetryRedactionTest, UncPathReducedToBasename) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("from \\\\server\\share\\dir\\weights.bin done"),
            "from weights.bin done");
}

TEST(TelemetryRedactionTest, UrlsArePreserved) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("see https://example.com/a/b/c for details"),
            "see https://example.com/a/b/c for details");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("ftp://host/path/file"), "ftp://host/path/file");
}

TEST(TelemetryRedactionTest, RelativePathsAndSlashesPreserved) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("models/foo.onnx"), "models/foo.onnx");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("a/b/c"), "a/b/c");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("ratio 3/4 and and/or"), "ratio 3/4 and and/or");
}

TEST(TelemetryRedactionTest, QuotedPath) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("file \"/home/alice/x/y.onnx\" missing"),
            "file \"y.onnx\" missing");
}

TEST(TelemetryRedactionTest, MultiplePaths) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("copy /home/u/a.onnx to /opt/cache/a.onnx"),
            "copy a.onnx to a.onnx");
}

TEST(TelemetryRedactionTest, BareRootsArePreserved) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("at / root"), "at / root");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("scheme-relative //host/x"), "scheme-relative //host/x");
}

TEST(TelemetryRedactionTest, TrailingPunctuationAfterPath) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("missing /home/alice/models/foo.onnx, retry"),
            "missing foo.onnx, retry");
}

TEST(TelemetryRedactionTest, HomeDirectoryReducedToTilde) {
  // A path that ends at the user's home directory must not emit the bare username.
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/home/alice"), "~");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/Users/alice"), "~");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("C:\\Users\\bob"), "~");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("could not access /home/alice"), "could not access ~");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/root"), "~");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("C:\\Users\\bob").find("bob"), std::string::npos);
}

TEST(TelemetryRedactionTest, DoesNotOverRedactUnrelatedHomeUsersDirs) {
  // A real file under a directory merely named home/users (not the first path component) is kept.
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/usr/home/config.txt"), "config.txt");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/opt/users/data.bin"), "data.bin");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/var/lib/users/cache.db"), "cache.db");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/home/alice/models/foo.onnx"), "foo.onnx");
}

TEST(TelemetryRedactionTest, PathsWithSpacesAreFullyReduced) {
  // The username and directory layout are dropped even when the path contains spaces.
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("Load C:\\Users\\First Last\\model.onnx failed"),
            "Load model.onnx failed");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("C:\\Program Files\\foo\\bar.dll"), "bar.dll");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("/Users/bob/Library/Application Support/x/m.onnx"),
            "m.onnx");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("Load C:\\Users\\First Last\\model.onnx failed").find("First"),
            std::string::npos);
}

TEST(TelemetryRedactionTest, PathsGluedToPunctuation) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("input:/home/alice/secret/m.onnx"), "input:m.onnx");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("paths /a/b/c.txt,/x/y/z.txt done"),
            "paths c.txt,z.txt done");
}

TEST(TelemetryRedactionTest, FileUriRedactedButHttpPreserved) {
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("file:///home/alice/secret/model.onnx"), "file:model.onnx");
  EXPECT_EQ(RedactAbsolutePathsForTelemetry("see https://example.com/a/b for x"),
            "see https://example.com/a/b for x");
}

}  // namespace test
}  // namespace onnxruntime
