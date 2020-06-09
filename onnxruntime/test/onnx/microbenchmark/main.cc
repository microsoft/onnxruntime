// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>
#include <core/graph/onnx_protobuf.h>
#include <core/common/logging/logging.h>
#include <core/platform/env.h>
#include <core/platform/threadpool.h>
#include <core/providers/cpu/cpu_execution_provider.h>
#include "core/session/environment.h"
#include <core/common/logging/sinks/clog_sink.h>
#include <core/platform/Barrier.h>
#include <core/graph/model.h>
#include <core/graph/graph.h>
#include <core/framework/kernel_def_builder.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/ort_env.h>
#include <core/util/thread_utils.h>

#include <unordered_map>

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
OrtEnv* env = nullptr;

using namespace onnxruntime;

static void BM_CPUAllocator(benchmark::State& state) {
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  const size_t len = state.range(0);
  for (auto _ : state) {
    void* p = cpu_allocator->Alloc(len);
    cpu_allocator->Free(p);
  }
}
BENCHMARK(BM_CPUAllocator)->Arg(4)->Arg(sizeof(Tensor));

static void BM_ResolveGraph(benchmark::State& state) {
  std::shared_ptr<onnxruntime::Model> model_copy;
  auto logger = env->GetLoggingManager()->CreateLogger("test");
  auto st =
      onnxruntime::Model::Load(ORT_TSTR("../models/opset8/test_tiny_yolov2/model.onnx"), model_copy, nullptr, *logger);
  if (!st.IsOK()) {
    printf("Parse model failed: %s", st.ErrorMessage().c_str());
    abort();
  }
  auto proto = model_copy->ToProto();
  model_copy.reset();
  for (auto _ : state) {
    state.PauseTiming();
    std::shared_ptr<onnxruntime::Model> model = std::make_shared<onnxruntime::Model>(proto, nullptr, *logger);
    onnxruntime::Graph& graph = model->MainGraph();
    state.ResumeTiming();
    st = graph.Resolve();
    if (!st.IsOK()) {
      printf("Resolve graph failed: %s", st.ErrorMessage().c_str());
      abort();
    }
  }
}

BENCHMARK(BM_ResolveGraph);
#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);


int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return -1;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "test", &env));
  ::benchmark::RunSpecifiedBenchmarks();
  g_ort->ReleaseEnv(env);
  return 0;
}
