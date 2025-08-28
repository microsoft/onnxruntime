// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"

#include "test/util/include/scoped_env_vars.h"
#include "test/common/trt_op_test_utils.h"
#include "test/common/random_generator.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <thread>
#include <chrono>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
extern std::unique_ptr<Ort::Env> ort_env;
namespace onnxruntime {

namespace test {

TEST(NvExecutionProviderTest, RuntimeCaching) {
  PathString model_name = ORT_TSTR("nv_execution_provider_runtime_caching.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_runtime_caching_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, 3, 2};
  std::string runtime_cahe_name = "runtime_cache.trt";
  CreateBaseModel(model_name, graph_name, dims);
  // AOT time
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {
      {"nv_runtime_cache_path", runtime_cahe_name.c_str()}
    });
    Ort::Session session_object(*ort_env, model_name.c_str(), so);

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }
  // the cache will be dumped to disk upon session destruction
  ASSERT_TRUE(std::filesystem::exists(runtime_cahe_name.c_str()));

  // use existing cache
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(*ort_env, model_name_ctx.c_str(), so);

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // create new cache
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    std::string new_cache_name = "runtime_cache_new.trt";
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {
      {"nv_runtime_cache_path", new_cache_name.c_str()}
    });
    {
      Ort::Session session_object(*ort_env, model_name_ctx.c_str(), so);
      auto io_binding = generate_io_binding(session_object);
      session_object.Run(run_options, io_binding);
    }
    // the cache will be dumped to disk upon session destruction
    ASSERT_TRUE(std::filesystem::exists(new_cache_name.c_str()));
  }

  // overwrite corrupted cache
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    // we assume the ONNX file as cache would should fail deserialization and overwrite the model
    std::string new_cache_name = PathToUTF8(model_name);
    auto file_size_old = std::filesystem::file_size(new_cache_name);
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {
      {"nv_runtime_cache_path", new_cache_name.c_str()}
    });
    {
      Ort::Session session_object(*ort_env, model_name_ctx.c_str(), so);
      auto io_binding = generate_io_binding(session_object);
      session_object.Run(run_options, io_binding);
    }
    auto file_size_new = std::filesystem::file_size(new_cache_name);
    // the cache will be dumped to disk upon session destruction
    ASSERT_TRUE(std::filesystem::exists(new_cache_name.c_str()));
    ASSERT_TRUE(file_size_old != file_size_new);
  }
}
}  // namespace test
}  // namespace onnxruntime
