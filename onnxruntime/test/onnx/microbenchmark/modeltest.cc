// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>
#include <core/graph/model.h>

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
