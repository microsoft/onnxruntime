// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>
#include <core/graph/model.h>
#include <core/platform/path_lib.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/ort_env.h>

#include "providers.h"

extern OrtEnv* env;
extern const OrtApi* g_ort;

static void BM_LoadModel(benchmark::State& state) {
  auto logger = env->GetLoggingManager()->CreateLogger("test");
  for (auto _ : state) {
    std::shared_ptr<onnxruntime::Model> yolomodel;
    auto st =
        onnxruntime::Model::Load(ORT_TSTR("../models/opset8/test_tiny_yolov2/model.onnx"), yolomodel, nullptr, *logger);
    if (!st.IsOK()) {
      state.SkipWithError(st.ErrorMessage().c_str());
      break;
    }
  }
}

BENCHMARK(BM_LoadModel);

#define ORT_BREAK_ON_ERROR(expr)                                \
  do {                                                          \
    OrtStatus* onnx_status = (expr);                            \
    if (onnx_status != NULL) {                                  \
      state.SkipWithError(g_ort->GetErrorMessage(onnx_status)); \
      g_ort->ReleaseStatus(onnx_status);                        \
    }                                                           \
  } while (0);

#ifdef USE_CUDA
static void BM_CreateSession_WithGPU(benchmark::State& state) {
  const ORTCHAR_T* model_path = ORT_TSTR("../models/opset8/test_bvlc_alexnet/model.onnx");
  OrtSessionOptions* session_option;
  ORT_BREAK_ON_ERROR(g_ort->CreateSessionOptions(&session_option));
  ORT_BREAK_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
  for (auto _ : state) {
    OrtSession* session;
    ORT_BREAK_ON_ERROR(g_ort->CreateSession(env, model_path, session_option, &session));
    state.PauseTiming();
    g_ort->ReleaseSession(session);
    state.ResumeTiming();
  }
  g_ort->ReleaseSessionOptions(session_option);
}
BENCHMARK(BM_CreateSession_WithGPU);
#endif

static void BM_CreateSession(benchmark::State& state) {
  const ORTCHAR_T* model_path = ORT_TSTR("../models/opset8/test_bvlc_alexnet/model.onnx");
  OrtSessionOptions* session_option;
  ORT_BREAK_ON_ERROR(g_ort->CreateSessionOptions(&session_option));
  for (auto _ : state) {
    OrtSession* session;
    ORT_BREAK_ON_ERROR(g_ort->CreateSession(env, model_path, session_option, &session));
    state.PauseTiming();
    g_ort->ReleaseSession(session);
    state.ResumeTiming();
  }
  g_ort->ReleaseSessionOptions(session_option);
}
BENCHMARK(BM_CreateSession);
