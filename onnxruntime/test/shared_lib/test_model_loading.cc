// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#include <fstream>
#include "test_fixture.h"
#include "file_util.h"
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(CApiTest, model_from_array) {
  const char* model_path = "testdata/matmul_1.onnx";
  std::vector<char> buffer;
  {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file)
      ORT_THROW("Error reading model");
    buffer.resize(file.tellg());
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), buffer.size()))
      ORT_THROW("Error reading model");
  }

  Ort::SessionOptions so;
  Ort::Session session(*ort_env.get(), buffer.data(), buffer.size(), so);

#ifdef USE_CUDA
  // test with CUDA provider when using onnxruntime as dll
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(so, 0));
  Ort::Session session_cuda(*ort_env.get(), buffer.data(), buffer.size(), so);
#endif
}
}  // namespace test
}  // namespace onnxruntime
