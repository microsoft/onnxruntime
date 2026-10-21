// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_redaction.h"

#include <string>
#include <string_view>

#include "gtest/gtest.h"

namespace onnxruntime::test {

TEST(TelemetryRedactionTest, RedactsPathsToEndOfMessage) {
  const struct {
    const char* input;
    const char* expected;
  } cases[] = {
      {"/home/alice/model.onnx", "[path]"},
      {"/data/models/secret/", "[path]"},
      {"~/.config/app/x", "[path]"},
      {"~\\config\\app", "[path]"},
      {"C:\\Users\\bob\\model.onnx", "[path]"},
      {"C:/Users\\bob\\model.onnx", "[path]"},
      {"\\\\server\\share\\dir\\weights.bin", "[path]"},
      {"Load model from /home/alice/models/foo.onnx failed", "Load model from [path]"},
      {"Load C:\\proj\\bin\\m.onnx failed", "Load [path]"},
      {"open D:/data/secret/model.onnx", "open [path]"},
      {"from \\\\server\\share\\dir\\weights.bin done", "from [path]"},
      {"Load C:\\Users\\First Last\\model.onnx failed", "Load [path]"},
      {"C:\\UsErS\\alice\\model.onnx", "[path]"},
      {"C:\\Users/alice\\model.onnx", "[path]"},
      {"/UsErS/alice/model.onnx", "[path]"},
      {"C:\\Users/alice/proj\\model.onnx", "[path]"},
      {"/home//alice/model.onnx", "[path]"},
      {"C:\\Users\\\\alice\\model.onnx", "[path]"},
      {"/home/./alice/model.onnx", "[path]"},
      {"at proj\\alice\\weights\\m.onnx", "at [path]"},
      {"alice/models/phi3.onnx", "[path]"},
      {"at alice/models/phi3.onnx", "at [path]"},
      {"a/b/c", "[path]"},
      {"x/y/z/", "[path]"},
      {"a\\b\\c", "[path]"},
      {"alice\\models\\phi3.onnx", "[path]"},
      {"Users\\alice\\model.onnx", "[path]"},
      {"Load Users\\bob\\m.onnx failed", "Load [path]"},
      {"Users\\First Last\\model.onnx", "[path]"},
      {"Load Users\\First Last\\m.onnx failed", "Load [path]"},
  };
  for (const auto& [input, expected] : cases) {
    SCOPED_TRACE(input);
    EXPECT_EQ(ScrubStringForTelemetry(input), expected);
    EXPECT_EQ(ScrubStringForTelemetry(std::string_view(input)), expected);
  }
  // URI schemes can resemble drive prefixes; their retained prefix is not part of the privacy contract.
  for (const char* input : {"input:/home/alice/secret/m.onnx", "file:///home/alice/secret/model.onnx"}) {
    SCOPED_TRACE(input);
    const auto output = ScrubStringForTelemetry(input);
    EXPECT_EQ(output.find("alice"), std::string::npos);
    EXPECT_NE(output.find("[path]"), std::string::npos);
  }
}

TEST(TelemetryRedactionTest, PreservesNonPaths) {
  for (const char* input : {"", "no path here", "error code 13", "models/foo.onnx",
                            "ratio 3/4 and/or", "domain\\user", "read\\write access"}) {
    SCOPED_TRACE(input);
    EXPECT_EQ(ScrubStringForTelemetry(input), input);
  }
}

TEST(TelemetryRedactionTest, BoundsAndSanitizesOutput) {
  EXPECT_EQ(ScrubStringForTelemetry(std::string(kMaxTelemetryStringLength + 1, 'x')),
            std::string(kMaxTelemetryStringLength, 'x'));
  const std::string euro = "\xE2\x82\xAC";
  const std::string prefix(kMaxTelemetryStringLength - 1, 'x');
  EXPECT_EQ(ScrubStringForTelemetry(prefix + euro), prefix);
  const std::string exact = std::string(kMaxTelemetryStringLength - euro.size(), 'x') + euro;
  EXPECT_EQ(ScrubStringForTelemetry(exact), exact);
  EXPECT_EQ(ScrubStringForTelemetry("\x80\xC0\xAF"), "???");
}

TEST(TelemetryRedactionTest, PreservesPathEvidenceAcrossIntakeLimits) {
  for (const char separator : {'/', '\\'}) {
    const std::string path = "Load alice" + std::string(1, separator) +
                             std::string(2000, 'x') + separator + "model";
    EXPECT_EQ(ScrubStringForTelemetry(path), "Load [path]");
    const std::string oversized = "alice" + std::string(1, separator) +
                                  std::string(telemetry_detail::kMaxTelemetryProbeBytes, 'x') +
                                  separator + "model";
    const auto view = telemetry_detail::TelemetryCStringView(
        oversized.c_str(), telemetry_detail::kMaxTelemetryProbeBytes);
    EXPECT_EQ(view.size(), telemetry_detail::kMaxTelemetryProbeBytes + 1);
    EXPECT_EQ(ScrubStringForTelemetry(view), "[path]");
    EXPECT_EQ(ScrubStringForTelemetry(oversized.c_str()), "[path]");
  }
  EXPECT_EQ(ScrubStringForTelemetry(std::string(telemetry_detail::kMaxTelemetryProbeBytes + 1, 'a')), "[path]");
  EXPECT_EQ(ScrubStringForTelemetry("Load alice" + std::string(telemetry_detail::kMaxTelemetryProbeBytes, 'x') +
                                    "/models/file"),
            "Load [path]");
  EXPECT_EQ(ScrubStringForTelemetry("Load alice", true), "Load [path]");
  EXPECT_EQ(ScrubStringForTelemetry(nullptr), "");
  const std::string harmless = "ratio 3/4 " + std::string(2000, 'x');
  EXPECT_EQ(ScrubStringForTelemetry(harmless), harmless.substr(0, kMaxTelemetryStringLength));
}

TEST(TelemetryRedactionTest, ScrubsBeforePointerStorageAndLocationTruncation) {
  const std::string message = "Using backend path: C:\\Users\\First Last\\QnnHtp.dll";
  telemetry_detail::TelemetryStrings strings;
  const char* stored_message = strings.Utf8(ScrubStringForTelemetry(message));
  const char* stored_identifier = strings.Utf8(ScrubStringForTelemetry("\\\\server\\share\\model.bin"));
  const char* stored_category = strings.Utf8(ScrubStringForTelemetry("backend /home/alice/qnn"));
  EXPECT_STREQ(stored_message, "Using backend path: [path]");
  EXPECT_STREQ(stored_identifier, "[path]");
  EXPECT_STREQ(stored_category, "backend [path]");
  EXPECT_EQ(message, "Using backend path: C:\\Users\\First Last\\QnnHtp.dll");

  const std::string function = "function alice/" + std::string(kMaxTelemetryStringLength, 'x') + "/model";
  std::string location = "qnn_execution_provider.cc:377 ";
  telemetry_detail::AppendTelemetryString(location, ScrubStringForTelemetry(function));
  EXPECT_EQ(location, "qnn_execution_provider.cc:377 function [path]");
}

}  // namespace onnxruntime::test
