// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>
#include <core/graph/onnx_protobuf.h>
#include <core/common/logging/logging.h>
#include <core/platform/env.h>
#include <core/providers/cpu/cpu_execution_provider.h>
#include <core/framework/environment.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/graph/model.h>
#include <core/graph/graph.h>
#include <core/framework/kernel_def_builder.h>
#include <unordered_map>

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
  auto st = onnxruntime::Model::Load("../models/opset8/test_tiny_yolov2/model.onnx", model_copy);
  if (!st.IsOK()) {
    printf("Parse model failed: %s", st.ErrorMessage().c_str());
    abort();
  }
  auto proto = model_copy->ToProto();
  model_copy.reset();
  for (auto _ : state) {
    state.PauseTiming();
    std::shared_ptr<onnxruntime::Model> model = std::make_shared<onnxruntime::Model>(proto);
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

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return -1;
  std::string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING, false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  std::unique_ptr<Environment> env;
  auto status = Environment::Create(env);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
