// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Verifies that the WebGPU plugin EP's self-contained path redaction stays equivalent to ONNX Runtime's.
// This is the only test that couples the plugin telemetry to ONNX Runtime internals; delete it when the plugin
// moves to its own repository.

#include <string>

#include <gtest/gtest.h>

#include "core/platform/telemetry_redaction.h"
#include "core/providers/webgpu/ep/telemetry_privacy.h"

namespace onnxruntime {
namespace test {

TEST(WebGpuPluginTelemetryTest, ScrubPathsMatchesOnnxRuntimeRedaction) {
  const std::string cases[] = {
      "",
      "plain message",
      "n/a and read/write are not paths",
      "domain\\user is not a path",
      "C:\\Users\\First Last\\model.onnx",
      "load failed for c:/a/b.onnx: reason",
      "\\\\server\\share\\x",
      "~/m.onnx and more",
      "~\\m.onnx",
      "/home/jane doe/models/m.onnx",
      "rel a/b/c.onnx",
      "rel alice\\models\\phi 3\\m.onnx",
      "\"quoted C:\\path\\x\"",
      "'single /a/b'",
      "line1\nC:\\x\\y",
      "bad utf8 \xC3 then \xFF",
      "multi byte \xE2\x82\xAC euro",
      std::string(1030, 'a') + " C:\\x\\y",
      std::string(1023, 'a') + "\xC3\xA9",
      std::string(17 * 1024, 'b'),
      std::string(100, 'c') + " " + std::string(17 * 1024, 'd'),
      std::string(100, 'e') + " /x" + std::string(17 * 1024, 'f'),
  };
  for (const auto& message : cases) {
    SCOPED_TRACE(message.substr(0, 64));
    EXPECT_EQ(webgpu::ep::telemetry::ScrubPaths(message), ScrubStringForTelemetry(message));
  }
}

}  // namespace test
}  // namespace onnxruntime
