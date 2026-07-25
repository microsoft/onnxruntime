// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT.
//
// These tests live in onnxruntime_test_all (static-lib linkage) so that future tests can
// call compile_utils.cc functions directly (e.g. epctx::BuildAndSaveOptimizedModel with a
// hand-constructed Model) without going through the full CompileModel path.
//
// Current tests exercise the flag end-to-end via the public Ort::ModelCompilationOptions
// C++ wrappers — the same API a caller would use — and inspect the output ONNX proto to
// verify the behavioural contract:
//   1. No EPContext nodes in the output (pure ONNX, not EPContext format).
//   2. Graph-level optimizations are applied (MaxLevel by default, not ORT_DISABLE_ALL).
//   3. All output targets (file, buffer, write-func) work correctly.
//   4. The resulting model can be re-loaded for inference with ORT_DISABLE_ALL.

#include <filesystem>
#include <fstream>
#include <string_view>

#include <gsl/gsl>
#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/util/include/api_asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

namespace {

void LoadModelProtoFromFile(const ORTCHAR_T* model_file, ONNX_NAMESPACE::ModelProto& model_proto) {
  std::ifstream model_stream{std::filesystem::path(model_file), std::ios::binary};
  ASSERT_TRUE(model_stream.is_open()) << "Failed to open: " << std::filesystem::path(model_file);
  ASSERT_TRUE(model_proto.ParseFromIstream(&model_stream)) << "Failed to parse model proto";
}

bool HasEpContextNodes(const ONNX_NAMESPACE::ModelProto& model_proto) {
  for (const auto& node : model_proto.graph().node()) {
    if (node.domain() == kMSDomain && node.op_type() == "EPContext") {
      return true;
    }
  }
  return false;
}

int CountNodesByOpType(const ONNX_NAMESPACE::ModelProto& model_proto, std::string_view op_type) {
  int count = 0;
  for (const auto& node : model_proto.graph().node()) {
    if (node.op_type() == op_type) {
      ++count;
    }
  }
  return count;
}

}  // namespace

// Tests that OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT saves a fully-optimized plain ONNX model
// (no EPContext nodes) to a file, and the model can be re-loaded for inference.
TEST(CompileApiOptimizedOnnxOutput, ToFile) {
  const ORTCHAR_T* input_model_path = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* output_model_path = ORT_TSTR("optimized_onnx_output.tofile.onnx");
  std::filesystem::remove(output_model_path);
  auto cleanup = gsl::finally([&]() { std::filesystem::remove(output_model_path); });

  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
  compile_options.SetFlags(OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT);
  compile_options.SetInputModelPath(input_model_path);
  compile_options.SetOutputModelPath(output_model_path);

  ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
  ASSERT_TRUE(std::filesystem::exists(output_model_path));

  // Output must be a valid ONNX model with no EPContext nodes.
  ONNX_NAMESPACE::ModelProto output_model;
  ASSERT_NO_FATAL_FAILURE(LoadModelProtoFromFile(output_model_path, output_model));
  EXPECT_FALSE(HasEpContextNodes(output_model));

  // Must be usable for inference with all graph optimizations disabled.
  Ort::SessionOptions reload_opts;
  reload_opts.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  EXPECT_NO_THROW(Ort::Session(*ort_env, output_model_path, reload_opts));
}

// Tests that OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT saves the optimized model to an
// in-memory buffer and the buffer contains a valid plain ONNX model (no EPContext nodes).
// This covers the sandboxed-compiler / GPU-process use case where no filesystem is available.
TEST(CompileApiOptimizedOnnxOutput, ToBuffer) {
  const ORTCHAR_T* input_model_path = ORT_TSTR("testdata/mul_1.onnx");

  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
  compile_options.SetFlags(OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT);
  compile_options.SetInputModelPath(input_model_path);

  Ort::AllocatorWithDefaultOptions allocator;
  void* output_buffer = nullptr;
  size_t output_size = 0;
  compile_options.SetOutputModelBuffer(allocator, &output_buffer, &output_size);
  auto cleanup = gsl::finally([&]() {
    if (output_buffer) allocator.Free(output_buffer);
  });

  ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
  ASSERT_NE(output_buffer, nullptr);
  ASSERT_GT(output_size, 0u);

  // Buffer must parse as a valid ONNX model with no EPContext nodes.
  ONNX_NAMESPACE::ModelProto output_model;
  ASSERT_TRUE(output_model.ParseFromArray(output_buffer, static_cast<int>(output_size)));
  EXPECT_FALSE(HasEpContextNodes(output_model));

  // Must be loadable for inference from the raw bytes.
  Ort::SessionOptions reload_opts;
  reload_opts.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  EXPECT_NO_THROW(Ort::Session(*ort_env, output_buffer, output_size, reload_opts));
}

// Tests that OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT can write the optimized model to
// a user-provided write function (stream), and the result is a valid plain ONNX model.
TEST(CompileApiOptimizedOnnxOutput, ToWriteFunc) {
  const ORTCHAR_T* input_model_path = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* output_model_path = ORT_TSTR("optimized_onnx_output.tostream.onnx");
  std::filesystem::remove(output_model_path);
  auto cleanup = gsl::finally([&]() { std::filesystem::remove(output_model_path); });

  struct WriteState {
    std::ofstream stream;
    size_t bytes_written = 0;
  };

  WriteState write_state{std::ofstream{std::filesystem::path(output_model_path), std::ios::binary}};
  ASSERT_TRUE(write_state.stream.is_open());

  auto write_func = [](void* state, const void* buffer, size_t size) -> OrtStatus* {
    auto* ws = static_cast<WriteState*>(state);
    ws->stream.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(size));
    ws->bytes_written += size;
    return nullptr;
  };

  {
    Ort::SessionOptions session_options;
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT);
    compile_options.SetInputModelPath(input_model_path);
    compile_options.SetOutputModelWriteFunc(write_func, &write_state);

    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
  }

  write_state.stream.flush();
  write_state.stream.close();

  EXPECT_GT(write_state.bytes_written, 0u);
  ASSERT_TRUE(std::filesystem::exists(output_model_path));

  // Written bytes must parse as a valid ONNX model with no EPContext nodes.
  ONNX_NAMESPACE::ModelProto output_model;
  ASSERT_NO_FATAL_FAILURE(LoadModelProtoFromFile(output_model_path, output_model));
  EXPECT_FALSE(HasEpContextNodes(output_model));
}

#if !defined(DISABLE_CONTRIB_OPS)
// Tests that OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT defaults to MaxLevel (ORT_ENABLE_ALL)
// graph optimizations when the caller does not set an explicit level, unlike the standard
// Compile API path which defaults to ORT_DISABLE_ALL.
//
// Uses conv_relu.onnx (Conv → Relu). ConvActivationFusion (Level 2, ORT_ENABLE_EXTENDED)
// merges them into a single com.microsoft.FusedConv node under MaxLevel but leaves them as
// separate ONNX ops under ORT_DISABLE_ALL, giving a clear Level-2-specific signal.
TEST(CompileApiOptimizedOnnxOutput, AppliesMaxLevelOptimizations) {
  const ORTCHAR_T* input_model_path = ORT_TSTR("testdata/transform/fusion/conv_relu.onnx");
  const ORTCHAR_T* output_no_opt_path = ORT_TSTR("optimized_onnx_output.conv_relu_no_opt.onnx");
  const ORTCHAR_T* output_maxlevel_path = ORT_TSTR("optimized_onnx_output.conv_relu_maxlevel.onnx");
  std::filesystem::remove(output_no_opt_path);
  std::filesystem::remove(output_maxlevel_path);
  auto cleanup = gsl::finally([&]() {
    std::filesystem::remove(output_no_opt_path);
    std::filesystem::remove(output_maxlevel_path);
  });

  // Default Compile API (no flag): uses ORT_DISABLE_ALL — Conv and Relu remain separate.
  {
    Ort::SessionOptions session_options;
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(input_model_path);
    compile_options.SetOutputModelPath(output_no_opt_path);
    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
  }

  // With OPTIMIZED_ONNX_OUTPUT flag: MaxLevel applies ConvActivationFusion (Level 2).
  {
    Ort::SessionOptions session_options;
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_OPTIMIZED_ONNX_OUTPUT);
    compile_options.SetInputModelPath(input_model_path);
    compile_options.SetOutputModelPath(output_maxlevel_path);
    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
  }

  ONNX_NAMESPACE::ModelProto model_no_opt;
  ONNX_NAMESPACE::ModelProto model_maxlevel;
  ASSERT_NO_FATAL_FAILURE(LoadModelProtoFromFile(output_no_opt_path, model_no_opt));
  ASSERT_NO_FATAL_FAILURE(LoadModelProtoFromFile(output_maxlevel_path, model_maxlevel));

  // Unoptimized baseline: Relu is present as a separate node (ORT_DISABLE_ALL).
  EXPECT_GT(CountNodesByOpType(model_no_opt, "Relu"), 0);
  // MaxLevel: ConvActivationFusion (Level 2) absorbs Relu into FusedConv.
  EXPECT_EQ(CountNodesByOpType(model_maxlevel, "Relu"), 0)
      << "OPTIMIZED_ONNX_OUTPUT should eliminate Relu via ConvActivationFusion (Level 2)";
  EXPECT_GT(CountNodesByOpType(model_maxlevel, "FusedConv"), 0)
      << "OPTIMIZED_ONNX_OUTPUT should produce FusedConv via Level 2 optimization";
}
#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
