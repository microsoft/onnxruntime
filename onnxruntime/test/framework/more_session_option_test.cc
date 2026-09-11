// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <filesystem>
#include <string>

#include "gtest/gtest.h"

#include "core/framework/session_options.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace test {

namespace {

// Tiny RAII wrapper around OrtStatus* so tests don't leak on failure.
struct OrtStatusGuard {
  OrtStatus* st = nullptr;
  ~OrtStatusGuard() {
    if (st != nullptr) Ort::GetApi().ReleaseStatus(st);
  }
};

const OrtApi& Api() { return Ort::GetApi(); }

// Build a fresh OrtSessionOptions* via the C API for tests that need a raw handle.
OrtSessionOptions* MakeOptions() {
  OrtSessionOptions* opts = nullptr;
  OrtStatus* st = Api().CreateSessionOptions(&opts);
  EXPECT_EQ(st, nullptr);
  if (st != nullptr) Api().ReleaseStatus(st);
  return opts;
}

void ReleaseOptions(OrtSessionOptions* opts) { Api().ReleaseSessionOptions(opts); }

OrtStatus* AddOption(OrtSessionOptions* opts, const char* key, const char* value) {
  return Api().AddSessionConfigEntry(opts, key, value);
}

OrtErrorCode CodeOf(OrtStatus* st) { return Api().GetErrorCode(st); }
const char* MsgOf(OrtStatus* st) { return Api().GetErrorMessage(st); }

}  // namespace

// -----------------------------------------------------------------------------
// Bool-valued keys: session.enable_cpu_mem_arena, session.enable_mem_pattern,
//                   session.use_deterministic_compute
// -----------------------------------------------------------------------------
TEST(CApiTest, BoolKeys_AcceptsAllSpellings) {
  // ParseBool accepts: "0", "1", "true", "false", and case-insensitive variants.
  const char* truthy[] = {"1", "true", "True", "TRUE"};
  const char* falsy[] = {"0", "false", "False", "FALSE"};

  for (const char* v : truthy) {
    OrtSessionOptions* opts = MakeOptions();
    OrtStatusGuard g{AddOption(opts, "session.enable_cpu_mem_arena", v)};
    ASSERT_EQ(g.st, nullptr) << "value=" << v;
    EXPECT_TRUE(opts->value.enable_cpu_mem_arena) << "value=" << v;
    ReleaseOptions(opts);
  }
  for (const char* v : falsy) {
    OrtSessionOptions* opts = MakeOptions();
    OrtStatusGuard g{AddOption(opts, "session.enable_cpu_mem_arena", v)};
    ASSERT_EQ(g.st, nullptr) << "value=" << v;
    EXPECT_FALSE(opts->value.enable_cpu_mem_arena) << "value=" << v;
    ReleaseOptions(opts);
  }
}

TEST(CApiTest, BoolKey_EnableMemPattern) {
  OrtSessionOptions* opts = MakeOptions();
  ASSERT_TRUE(opts->value.enable_mem_pattern);  // default

  {
    OrtStatusGuard g{AddOption(opts, "session.enable_mem_pattern", "0")};
    ASSERT_EQ(g.st, nullptr);
    EXPECT_FALSE(opts->value.enable_mem_pattern);
  }
  {
    OrtStatusGuard g{AddOption(opts, "session.enable_mem_pattern", "true")};
    ASSERT_EQ(g.st, nullptr);
    EXPECT_TRUE(opts->value.enable_mem_pattern);
  }
  ReleaseOptions(opts);
}

TEST(CApiTest, BoolKey_UseDeterministicCompute) {
  OrtSessionOptions* opts = MakeOptions();
  EXPECT_FALSE(opts->value.use_deterministic_compute);
  OrtStatusGuard g{AddOption(opts, "session.use_deterministic_compute", "1")};
  ASSERT_EQ(g.st, nullptr);
  EXPECT_TRUE(opts->value.use_deterministic_compute);
  ReleaseOptions(opts);
}

TEST(CApiTest, BoolKey_InvalidValueErrors) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.enable_cpu_mem_arena", "yes")};
  ASSERT_NE(g.st, nullptr);
  EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(MsgOf(g.st)).find("boolean"), std::string::npos) << MsgOf(g.st);
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// Int-valued keys: session.intra_op_num_threads, session.inter_op_num_threads,
//                  session.log_severity_level, session.log_verbosity_level
// -----------------------------------------------------------------------------
TEST(CApiTest, IntKey_IntraOpNumThreads) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.intra_op_num_threads", "4")};
  ASSERT_EQ(g.st, nullptr);
  EXPECT_EQ(opts->value.intra_op_param.thread_pool_size, 4);
  ReleaseOptions(opts);
}

TEST(CApiTest, IntKey_InterOpNumThreads) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.inter_op_num_threads", "2")};
  ASSERT_EQ(g.st, nullptr);
  EXPECT_EQ(opts->value.inter_op_param.thread_pool_size, 2);
  ReleaseOptions(opts);
}

TEST(CApiTest, IntKey_LogSeverityAndVerbosity) {
  OrtSessionOptions* opts = MakeOptions();
  {
    OrtStatusGuard g{AddOption(opts, "session.log_severity_level", "2")};
    ASSERT_EQ(g.st, nullptr);
    EXPECT_EQ(opts->value.session_log_severity_level, 2);
  }
  {
    OrtStatusGuard g{AddOption(opts, "session.log_verbosity_level", "3")};
    ASSERT_EQ(g.st, nullptr);
    EXPECT_EQ(opts->value.session_log_verbosity_level, 3);
  }
  ReleaseOptions(opts);
}

TEST(CApiTest, IntKey_EmptyValueErrors) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.intra_op_num_threads", "")};
  ASSERT_NE(g.st, nullptr);
  EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(MsgOf(g.st)).find("empty"), std::string::npos) << MsgOf(g.st);
  ReleaseOptions(opts);
}

TEST(CApiTest, IntKey_NonIntegerValueErrors) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.intra_op_num_threads", "not_an_int")};
  ASSERT_NE(g.st, nullptr);
  EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(MsgOf(g.st)).find("base-10 int32"), std::string::npos) << MsgOf(g.st);
  ReleaseOptions(opts);
}

TEST(CApiTest, IntKey_TrailingGarbageErrors) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.intra_op_num_threads", "12abc")};
  ASSERT_NE(g.st, nullptr);
  EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// Log id (session.log_id)
// -----------------------------------------------------------------------------
TEST(CApiTest, LogId_SetViaConfigEntry) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.log_id", "session-A")};
  ASSERT_EQ(g.st, nullptr);
  EXPECT_EQ(opts->value.session_logid, "session-A");
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// Enable profiling: non-empty enables with prefix, empty disables.
// -----------------------------------------------------------------------------
TEST(CApiTest, EnableProfiling_NonEmptyEnables) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.enable_profiling", "myrun_")};
  ASSERT_EQ(g.st, nullptr);
  EXPECT_TRUE(opts->value.enable_profiling);
  EXPECT_EQ(opts->value.profile_file_prefix, ORT_TSTR("myrun_"));
  ReleaseOptions(opts);
}

TEST(CApiTest, EnableProfiling_EmptyDisables) {
  OrtSessionOptions* opts = MakeOptions();
  // First enable.
  {
    OrtStatusGuard g{AddOption(opts, "session.enable_profiling", "x_")};
    ASSERT_EQ(g.st, nullptr);
    ASSERT_TRUE(opts->value.enable_profiling);
  }
  // Empty string disables.
  {
    OrtStatusGuard g{AddOption(opts, "session.enable_profiling", "")};
    ASSERT_EQ(g.st, nullptr);
    EXPECT_FALSE(opts->value.enable_profiling);
  }
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// Graph optimization level (enum)
// -----------------------------------------------------------------------------
TEST(CApiTest, GraphOptimizationLevel_AllSpellings) {
  struct Case {
    const char* in;
    TransformerLevel expected;
  };
  const Case cases[] = {
      {"disable_all", TransformerLevel::Default},
      {"enable_basic", TransformerLevel::Level1},
      {"enable_extended", TransformerLevel::Level2},
      {"enable_layout", TransformerLevel::Level3},
      {"enable_all", TransformerLevel::MaxLevel},
      {"ENABLE_ALL", TransformerLevel::MaxLevel},
      {"Enable_Basic", TransformerLevel::Level1},
  };
  for (const auto& c : cases) {
    OrtSessionOptions* opts = MakeOptions();
    OrtStatusGuard g{AddOption(opts, "session.graph_optimization_level", c.in)};
    ASSERT_EQ(g.st, nullptr) << "value=" << c.in;
    EXPECT_EQ(opts->value.graph_optimization_level, c.expected) << "value=" << c.in;
    ReleaseOptions(opts);
  }
}

TEST(CApiTest, GraphOptimizationLevel_InvalidValueErrors) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.graph_optimization_level", "ludicrous")};
  ASSERT_NE(g.st, nullptr);
  EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(MsgOf(g.st)).find("graph_optimization_level"), std::string::npos);
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// Optimized model filepath (path)
// -----------------------------------------------------------------------------
TEST(CApiTest, OptimizedModelFilepath) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.optimized_model_filepath", "/tmp/opt_model.onnx")};
  ASSERT_EQ(g.st, nullptr);
  EXPECT_EQ(opts->value.optimized_model_filepath,
            std::filesystem::path(ORT_TSTR("/tmp/opt_model.onnx")));
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// Execution mode (enum)
// -----------------------------------------------------------------------------
TEST(CApiTest, ExecutionMode_SequentialAndParallel) {
  {
    OrtSessionOptions* opts = MakeOptions();
    OrtStatusGuard g{AddOption(opts, "session.execution_mode", "parallel")};
    ASSERT_EQ(g.st, nullptr);
    EXPECT_EQ(opts->value.execution_mode, ExecutionMode::ORT_PARALLEL);
    ReleaseOptions(opts);
  }
  {
    OrtSessionOptions* opts = MakeOptions();
    OrtStatusGuard g{AddOption(opts, "session.execution_mode", "Sequential")};
    ASSERT_EQ(g.st, nullptr);
    EXPECT_EQ(opts->value.execution_mode, ExecutionMode::ORT_SEQUENTIAL);
    ReleaseOptions(opts);
  }
}

TEST(CApiTest, ExecutionMode_InvalidValueErrors) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.execution_mode", "concurrent")};
  ASSERT_NE(g.st, nullptr);
  EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// use_per_session_threads — one-way disable, true is a no-op while still enabled,
// true after a disable is an error (no public ABI to re-enable).
// On WASM+pthreads the default is false, so setting "true" is always an error.
// -----------------------------------------------------------------------------
TEST(CApiTest, UsePerSessionThreads_TrueIsNoOpInitially) {
  OrtSessionOptions* opts = MakeOptions();
  constexpr bool kDefault = onnxruntime::SessionOptions::DEFAULT_USE_PER_SESSION_THREADS;
  ASSERT_EQ(opts->value.use_per_session_threads, kDefault);  // platform-specific default
  OrtStatusGuard g{AddOption(opts, "session.use_per_session_threads", "true")};
  if constexpr (kDefault) {
    // When default is true, setting "true" is a harmless no-op.
    ASSERT_EQ(g.st, nullptr);
    EXPECT_TRUE(opts->value.use_per_session_threads);
  } else {
    // When default is false (WASM+pthreads), setting "true" is an error
    // because there is no public ABI to re-enable.
    ASSERT_NE(g.st, nullptr);
    EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  }
  ReleaseOptions(opts);
}

TEST(CApiTest, UsePerSessionThreads_FalseDisables) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.use_per_session_threads", "false")};
  ASSERT_EQ(g.st, nullptr);
  EXPECT_FALSE(opts->value.use_per_session_threads);
  ReleaseOptions(opts);
}

TEST(CApiTest, UsePerSessionThreads_TrueAfterDisableErrors) {
  OrtSessionOptions* opts = MakeOptions();
  {
    OrtStatusGuard g{AddOption(opts, "session.use_per_session_threads", "false")};
    ASSERT_EQ(g.st, nullptr);
  }
  {
    OrtStatusGuard g{AddOption(opts, "session.use_per_session_threads", "true")};
    ASSERT_NE(g.st, nullptr);
    EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
    EXPECT_NE(std::string(MsgOf(g.st)).find("use_per_session_threads"), std::string::npos);
  }
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// Unknown keys are stored as regular session config entries.
// -----------------------------------------------------------------------------
TEST(CApiTest, UnknownKey_FallsThroughToConfigEntry) {
  OrtSessionOptions* opts = MakeOptions();
  OrtStatusGuard g{AddOption(opts, "session.disable_prepacking", "1")};
  ASSERT_EQ(g.st, nullptr);

  // Verify it landed in the config entries (not the typed setters).
  int has = 0;
  OrtStatusGuard g2{Api().HasSessionConfigEntry(opts, "session.disable_prepacking", &has)};
  ASSERT_EQ(g2.st, nullptr);
  EXPECT_EQ(has, 1);

  char buf[16] = {};
  size_t size = sizeof(buf);
  OrtStatusGuard g3{Api().GetSessionConfigEntry(opts, "session.disable_prepacking", buf, &size)};
  ASSERT_EQ(g3.st, nullptr);
  EXPECT_STREQ(buf, "1");
  ReleaseOptions(opts);
}

// -----------------------------------------------------------------------------
// Null arguments are rejected with INVALID_ARGUMENT.
// -----------------------------------------------------------------------------
TEST(CApiTest, NullArguments_AreRejected) {
  OrtSessionOptions* opts = MakeOptions();
  {
    OrtStatusGuard g{AddOption(nullptr, "session.intra_op_num_threads", "1")};
    ASSERT_NE(g.st, nullptr);
    EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  }
  {
    OrtStatusGuard g{AddOption(opts, nullptr, "1")};
    ASSERT_NE(g.st, nullptr);
    EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  }
  {
    OrtStatusGuard g{AddOption(opts, "session.intra_op_num_threads", nullptr)};
    ASSERT_NE(g.st, nullptr);
    EXPECT_EQ(CodeOf(g.st), ORT_INVALID_ARGUMENT);
  }
  ReleaseOptions(opts);
}

}  // namespace test
}  // namespace onnxruntime
