// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>
#include <core/graph/model.h>
#include <core/framework/path_lib.h>
#include <core/session/onnxruntime_c_api.h>
#include "providers.h"

static void BM_LoadModel(benchmark::State& state) {
  for (auto _ : state) {
    std::shared_ptr<onnxruntime::Model> yolomodel;
    auto st = onnxruntime::Model::Load("../models/opset8/test_tiny_yolov2/model.onnx", yolomodel);
    if (!st.IsOK()) {
      state.SkipWithError(st.ErrorMessage().c_str());
      break;
    }
  }
}

BENCHMARK(BM_LoadModel);

extern OrtEnv* env;

#define ORT_BREAK_ON_ERROR(expr)                            \
  do {                                                      \
    OrtStatus* onnx_status = (expr);                        \
    if (onnx_status != NULL) {                              \
      state.SkipWithError(OrtGetErrorMessage(onnx_status)); \
      OrtReleaseStatus(onnx_status);                        \
    }                                                       \
  } while (0);

#ifdef USE_CUDA
static void BM_CreateSession_WithGPU(benchmark::State& state) {
  const char* model_path = "../models/opset8/test_bvlc_alexnet/model.onnx";
  OrtSessionOptions* session_option = OrtCreateSessionOptions();
  ORT_BREAK_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
  for (auto _ : state) {
    OrtSession* session;
    ORT_BREAK_ON_ERROR(OrtCreateSession(env, model_path, session_option, &session));
    state.PauseTiming();
    OrtReleaseSession(session);
    state.ResumeTiming();
  }
  OrtReleaseSessionOptions(session_option);
}
BENCHMARK(BM_CreateSession_WithGPU);
#endif

static void BM_CreateSession(benchmark::State& state) {
  const ORTCHAR_T* model_path = ORT_TSTR("../models/opset8/test_bvlc_alexnet/model.onnx");
  OrtSessionOptions* session_option;
  ORT_BREAK_ON_ERROR(OrtCreateSessionOptions(&session_option));
  for (auto _ : state) {
    OrtSession* session;
    ORT_BREAK_ON_ERROR(OrtCreateSession(env, model_path, session_option, &session));
    state.PauseTiming();
    OrtReleaseSession(session);
    state.ResumeTiming();
  }
  OrtReleaseSessionOptions(session_option);
}
BENCHMARK(BM_CreateSession);
