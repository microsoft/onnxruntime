// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit and integration tests for the WebGPU EP's ProgramManager infrastructure.
//
// New WebGPU EP infrastructure tests (shader compilation diagnostics, program cache behavior,
// pipeline creation error handling, etc.) should be added here rather than in a separate file.

#include "gtest/gtest.h"

#include <array>
#include <string>
#include <string_view>

#include "core/providers/webgpu/program_manager_helpers.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_external_header.h"

#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

using onnxruntime::webgpu::detail::AnnotateShaderWithLineNumbers;
using onnxruntime::webgpu::detail::FormatShaderCompilationInfo;

namespace {

// Constructs a wgpu::CompilationMessage with the given fields. `message_text` must outlive the
// returned struct because StringView stores a pointer into it.
wgpu::CompilationMessage MakeMessage(std::string_view message_text,
                                     wgpu::CompilationMessageType type,
                                     uint64_t line_num,
                                     uint64_t line_pos,
                                     uint64_t length) {
  wgpu::CompilationMessage msg{};
  msg.message = wgpu::StringView{message_text.data(), message_text.size()};
  msg.type = type;
  msg.lineNum = line_num;
  msg.linePos = line_pos;
  msg.offset = 0;
  msg.length = length;
  return msg;
}

}  // namespace

// FormatShaderCompilationInfo returns an empty string when info is null or has no messages.
TEST(ShaderCompilationErrorFormatterTest, EmptyWhenNoMessages) {
  EXPECT_EQ(FormatShaderCompilationInfo("some code\n", nullptr), "");

  wgpu::CompilationInfo info{};
  info.messageCount = 0;
  info.messages = nullptr;
  EXPECT_EQ(FormatShaderCompilationInfo("some code\n", &info), "");
}

// Verifies the diagnostic header uses "error" for Error and includes line/column and message text,
// and that the source line and a caret with `~` tail spanning `length` are emitted.
TEST(ShaderCompilationErrorFormatterTest, ErrorWithCaretAndSourceLine) {
  const std::string code =
      "fn main() {\n"
      "  let x = foo();\n"
      "}\n";

  const std::string message_text = "unresolved identifier 'foo'";
  // line 2, column 11 (1-based), length 3 -> points at "foo".
  std::array<wgpu::CompilationMessage, 1> msgs{
      MakeMessage(message_text, wgpu::CompilationMessageType::Error, 2, 11, 3)};

  wgpu::CompilationInfo info{};
  info.messageCount = msgs.size();
  info.messages = msgs.data();

  std::string formatted = FormatShaderCompilationInfo(code, &info);

  // Header line: "  error at line 2:11: unresolved identifier 'foo'\n"
  EXPECT_NE(formatted.find("error at line 2:11: unresolved identifier 'foo'"), std::string::npos)
      << "formatted output was:\n"
      << formatted;

  // Source line with the "    | " prefix.
  EXPECT_NE(formatted.find("    |   let x = foo();"), std::string::npos)
      << "formatted output was:\n"
      << formatted;

  // Caret line: "    | " + 10 spaces + "^~~" (linePos=11 -> col=10; length=3 -> ^~~).
  const std::string expected_caret_line = "    |           ^~~";
  EXPECT_NE(formatted.find(expected_caret_line), std::string::npos)
      << "expected caret line: \"" << expected_caret_line << "\"\n"
      << "formatted output was:\n"
      << formatted;
}

// Warnings should be labeled "warning" (not "error").
TEST(ShaderCompilationErrorFormatterTest, WarningLabel) {
  const std::string code = "let unused = 1;\n";
  const std::string text = "unused variable";
  std::array<wgpu::CompilationMessage, 1> msgs{
      MakeMessage(text, wgpu::CompilationMessageType::Warning, 1, 5, 6)};

  wgpu::CompilationInfo info{};
  info.messageCount = msgs.size();
  info.messages = msgs.data();

  std::string formatted = FormatShaderCompilationInfo(code, &info);
  EXPECT_NE(formatted.find("warning at line 1:5: unused variable"), std::string::npos)
      << formatted;
  EXPECT_EQ(formatted.find("error at line"), std::string::npos)
      << "warning message was mis-labeled as error:\n"
      << formatted;
}

// Multiple messages should all be emitted, in order.
TEST(ShaderCompilationErrorFormatterTest, MultipleMessages) {
  const std::string code =
      "line one\n"
      "line two\n"
      "line three\n";

  const std::string t1 = "first";
  const std::string t2 = "second";
  std::array<wgpu::CompilationMessage, 2> msgs{
      MakeMessage(t1, wgpu::CompilationMessageType::Error, 1, 1, 4),
      MakeMessage(t2, wgpu::CompilationMessageType::Warning, 3, 6, 5)};

  wgpu::CompilationInfo info{};
  info.messageCount = msgs.size();
  info.messages = msgs.data();

  std::string formatted = FormatShaderCompilationInfo(code, &info);
  auto p1 = formatted.find("error at line 1:1: first");
  auto p2 = formatted.find("warning at line 3:6: second");
  ASSERT_NE(p1, std::string::npos) << formatted;
  ASSERT_NE(p2, std::string::npos) << formatted;
  EXPECT_LT(p1, p2) << "messages should appear in order:\n" << formatted;

  // Source lines for both messages should be present.
  EXPECT_NE(formatted.find("    | line one"), std::string::npos) << formatted;
  EXPECT_NE(formatted.find("    | line three"), std::string::npos) << formatted;
}

// A message pointing at a line that does not exist in the source should still emit the header,
// but must not print a spurious source line / caret block.
TEST(ShaderCompilationErrorFormatterTest, OutOfRangeLineOmitsSource) {
  const std::string code = "only one line\n";
  const std::string text = "bad";
  std::array<wgpu::CompilationMessage, 1> msgs{
      MakeMessage(text, wgpu::CompilationMessageType::Error, 42, 1, 1)};

  wgpu::CompilationInfo info{};
  info.messageCount = msgs.size();
  info.messages = msgs.data();

  std::string formatted = FormatShaderCompilationInfo(code, &info);
  EXPECT_NE(formatted.find("error at line 42:1: bad"), std::string::npos) << formatted;
  EXPECT_EQ(formatted.find("    |"), std::string::npos)
      << "no source-line block should be emitted for an out-of-range line:\n"
      << formatted;
}

// A zero-length span should still render a single caret ('^').
TEST(ShaderCompilationErrorFormatterTest, ZeroLengthProducesSingleCaret) {
  const std::string code = "abcdef\n";
  const std::string text = "point-in-line";
  std::array<wgpu::CompilationMessage, 1> msgs{
      MakeMessage(text, wgpu::CompilationMessageType::Error, 1, 4, 0)};

  wgpu::CompilationInfo info{};
  info.messageCount = msgs.size();
  info.messages = msgs.data();

  std::string formatted = FormatShaderCompilationInfo(code, &info);
  // linePos=4 -> col=3 -> 3 spaces then "^".
  EXPECT_NE(formatted.find("    |    ^\n"), std::string::npos)
      << "zero-length should render exactly one caret:\n"
      << formatted;
}

// AnnotateShaderWithLineNumbers should prefix every line with a right-aligned 5-wide number.
TEST(ShaderCompilationErrorFormatterTest, AnnotateShaderPrefixesLineNumbers) {
  const std::string code =
      "alpha\n"
      "beta\n"
      "gamma\n";

  std::string annotated = AnnotateShaderWithLineNumbers(code);
  EXPECT_EQ(annotated,
            "    1 | alpha\n"
            "    2 | beta\n"
            "    3 | gamma\n")
      << annotated;
}

// A file without a trailing newline should still emit the final line.
TEST(ShaderCompilationErrorFormatterTest, AnnotateShaderNoTrailingNewline) {
  const std::string code = "one\ntwo";
  EXPECT_EQ(AnnotateShaderWithLineNumbers(code),
            "    1 | one\n"
            "    2 | two\n");
}

// CR/LF line endings should not appear in the annotated output.
TEST(ShaderCompilationErrorFormatterTest, AnnotateShaderStripsCarriageReturns) {
  const std::string code = "one\r\ntwo\r\n";
  std::string annotated = AnnotateShaderWithLineNumbers(code);
  EXPECT_EQ(annotated,
            "    1 | one\n"
            "    2 | two\n")
      << annotated;
}

// The empty source should produce a single "line 1" entry with no content.
TEST(ShaderCompilationErrorFormatterTest, AnnotateShaderEmptyCode) {
  EXPECT_EQ(AnnotateShaderWithLineNumbers(""), "    1 | \n");
}

// End-to-end test that mimics the exact call chain used by `ProgramManager::Build` when a shader
// author writes invalid WGSL: real Dawn ShaderModule creation, real GetCompilationInfo callback,
// and our formatter. If a broken shader were placed in a real kernel, this is the metadata that
// would end up in the Status returned from `Build`.
TEST(ShaderCompilationErrorFormatterTest, EndToEnd_BrokenShaderProducesLineNumberMetadata) {
  // Ensure the default WebGPU context/device is initialized. If the WebGPU EP is unavailable
  // on this platform, skip the test.
  auto webgpu_ep = onnxruntime::test::DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  auto& webgpu_context = onnxruntime::webgpu::WebGpuContextFactory::DefaultContext();
  const auto& device = webgpu_context.Device();

  // Intentionally-broken WGSL: `let` requires an initializer, and `not_a_type` is not a type.
  // We craft the error so we know exactly which line the compiler must complain about (line 4).
  const std::string code =
      "@compute @workgroup_size(1)\n"                    // 1
      "fn main() {\n"                                    // 2
      "  var ok : u32 = 0u;\n"                           // 3
      "  let broken : not_a_type = 42;\n"                // 4  <- error line
      "  ok = ok + 1u;\n"                                // 5
      "}\n";                                             // 6

  wgpu::ShaderSourceWGSL wgsl_source{};
  wgsl_source.code = code.c_str();

  wgpu::ShaderModuleDescriptor descriptor{};
  descriptor.nextInChain = &wgsl_source;
  descriptor.label = "ShaderCompilationErrorFormatterTest.BrokenShader";

  auto shader_module = device.CreateShaderModule(&descriptor);

  // Synchronously request compilation info using the same pattern as `ProgramManager::Build`.
  wgpu::CompilationInfoRequestStatus request_status = wgpu::CompilationInfoRequestStatus::CallbackCancelled;
  size_t message_count = 0;
  bool saw_error = false;
  bool saw_error_on_line_4 = false;
  std::string formatted;

  auto wait_status = webgpu_context.Wait(shader_module.GetCompilationInfo(
      wgpu::CallbackMode::WaitAnyOnly,
      [&code, &request_status, &message_count, &saw_error, &saw_error_on_line_4, &formatted](
          wgpu::CompilationInfoRequestStatus status,
          wgpu::CompilationInfo const* info) noexcept {
        request_status = status;
        if (info != nullptr) {
          message_count = info->messageCount;
          for (size_t i = 0; i < info->messageCount; ++i) {
            const auto& m = info->messages[i];
            if (m.type == wgpu::CompilationMessageType::Error) {
              saw_error = true;
              if (m.lineNum == 4) {
                saw_error_on_line_4 = true;
              }
            }
          }
          formatted = FormatShaderCompilationInfo(code, info);
        }
      }));

  ASSERT_TRUE(wait_status.IsOK()) << wait_status.ErrorMessage();
  ASSERT_EQ(request_status, wgpu::CompilationInfoRequestStatus::Success);
  ASSERT_GT(message_count, 0u) << "Dawn reported no compilation messages for a broken shader.";
  EXPECT_TRUE(saw_error) << "Dawn did not report an error for a broken shader.";
  EXPECT_TRUE(saw_error_on_line_4)
      << "Expected an error on line 4 of the broken shader, formatted output:\n"
      << formatted;

  // The formatted diagnostic must contain the "error at line 4:" prefix and the annotated source
  // line for line 4. This validates that the metadata we surface to callers is correct.
  ASSERT_FALSE(formatted.empty());
  EXPECT_NE(formatted.find("error at line 4:"), std::string::npos)
      << "formatted output was:\n"
      << formatted;
  EXPECT_NE(formatted.find("    |   let broken : not_a_type = 42;"), std::string::npos)
      << "formatted output was:\n"
      << formatted;

  // Print for developer visibility when running the test manually.
  std::cout << "=== Formatted diagnostic for broken shader ===\n"
            << formatted
            << "==============================================\n";
}

}  // namespace test
}  // namespace onnxruntime
