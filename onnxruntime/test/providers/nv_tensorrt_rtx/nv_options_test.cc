// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/providers/nv_tensorrt_rtx/nv_provider_options.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"

#include "test/util/include/scoped_env_vars.h"
#include "test/common/trt_op_test_utils.h"
#include "test/common/random_generator.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <thread>
#include <chrono>
#include <fstream>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
extern std::unique_ptr<Ort::Env> ort_env;
namespace onnxruntime {

namespace test {
namespace {
void CreateSyntheticNvEPContextModel(const PathString& model_path, int64_t embed_mode) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(11);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("NvEPContextEngineRequiresOptIn");
  auto add_value_info = [](ONNX_NAMESPACE::ValueInfoProto* value_info, const char* name) {
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  };
  add_value_info(graph->add_input(), "input");
  add_value_info(graph->add_output(), "output");

  auto* node = graph->add_node();
  node->set_name("ep_context");
  node->set_op_type("EPContext");
  node->set_domain(kMSDomain);
  node->add_input("input");
  node->add_output("output");
  auto* embed_attr = node->add_attribute();
  embed_attr->set_name("embed_mode");
  embed_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  embed_attr->set_i(embed_mode);
  auto* context_attr = node->add_attribute();
  context_attr->set_name("ep_cache_context");
  context_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  context_attr->set_s("untrusted_engine");
  auto* source_attr = node->add_attribute();
  source_attr->set_name("source");
  source_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  source_attr->set_s(kNvTensorRTRTXExecutionProvider);

  std::ofstream output(model_path, std::ios::binary);
  ASSERT_TRUE(output.is_open());
  ASSERT_TRUE(model.SerializeToOstream(&output));
}
}  // namespace

TEST(NvExecutionProviderTest, InvalidEngineDeserializationOption) {
  Ort::SessionOptions so;
  EXPECT_THROW(so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider,
                                          {{"nv_engine_deserialization_enable", "2"}}),
               Ort::Exception);
}

TEST(NvExecutionProviderTest, EPContextEngineRequiresOptIn) {
  for (const int64_t embed_mode : {int64_t{0}, int64_t{1}}) {
    const PathString model_path = ToPathString(
        "nv_ep_context_engine_requires_opt_in_" + std::to_string(embed_mode) + ".onnx");
    CreateSyntheticNvEPContextModel(model_path, embed_mode);

    Ort::SessionOptions so;
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    try {
      Ort::Session session(*ort_env, model_path.c_str(), so);
      FAIL() << "Expected EPContext engine deserialization to require explicit opt-in";
    } catch (const Ort::Exception& ex) {
      EXPECT_NE(std::string(ex.what()).find("EPContext engine deserialization is disabled"), std::string::npos);
    }

    std::error_code ec;
    std::filesystem::remove(model_path, ec);
  }
}

TEST(NvExecutionProviderTest, EPContextInvalidEmbedModeRejected) {
  const PathString model_path = ToPathString("nv_ep_context_invalid_embed_mode.onnx");
  CreateSyntheticNvEPContextModel(model_path, 2);

  Ort::SessionOptions so;
  so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
  try {
    Ort::Session session(*ort_env, model_path.c_str(), so);
    FAIL() << "Expected invalid EPContext embed mode to be rejected";
  } catch (const Ort::Exception& ex) {
    EXPECT_NE(std::string(ex.what()).find("embed_mode must be 0 or 1"), std::string::npos);
  }

  std::error_code ec;
  std::filesystem::remove(model_path, ec);
}

size_t countFilesInDirectory(const std::string& dir_path) {
  return std::distance(std::filesystem::directory_iterator(dir_path), std::filesystem::directory_iterator{});
}

TEST(NvExecutionProviderTest, RuntimeCaching) {
  PathString model_name = ORT_TSTR("nv_execution_provider_runtime_caching.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_runtime_caching_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, 3, 2};
  std::string runtime_cache_name = "./runtime_cache/";
  if (std::filesystem::exists(runtime_cache_name)) {
    std::filesystem::remove_all(runtime_cache_name);
  }
  CreateBaseModel(model_name, graph_name, dims);
  // AOT time
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {{"nv_runtime_cache_path", runtime_cache_name.c_str()}});
    Ort::Session session_object(*ort_env, model_name.c_str(), so);

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }
  // the cache will be dumped to disk upon session destruction
  ASSERT_TRUE(std::filesystem::exists(runtime_cache_name.c_str()));
  ASSERT_TRUE(1 == countFilesInDirectory(runtime_cache_name));

  // use existing cache
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider,
                               {{"nv_runtime_cache_path", runtime_cache_name.c_str()},
                                {"nv_engine_deserialization_enable", "1"}});
    Ort::Session session_object(*ort_env, model_name_ctx.c_str(), so);
  }
  ASSERT_TRUE(1 == countFilesInDirectory(runtime_cache_name));

  // create new cache
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    std::string new_cache_name = "/tmp/runtime_cache_new/";
    if (std::filesystem::exists(new_cache_name)) {
      std::filesystem::remove_all(new_cache_name);
    }
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider,
                               {{"nv_runtime_cache_path", new_cache_name.c_str()},
                                {"nv_engine_deserialization_enable", "1"}});
    {
      Ort::Session session_object(*ort_env, model_name_ctx.c_str(), so);
    }
    // the cache will be dumped to disk upon session destruction
    ASSERT_TRUE(std::filesystem::exists(new_cache_name.c_str()));
    ASSERT_TRUE(1 == countFilesInDirectory(new_cache_name));
  }
}
}  // namespace test
}  // namespace onnxruntime
