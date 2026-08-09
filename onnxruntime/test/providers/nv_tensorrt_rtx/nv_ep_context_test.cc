// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/common/path_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <fstream>
#include <filesystem>

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

RegisteredEpDeviceUniquePtr AppendTrtEtxEP(Ort::SessionOptions& session_options, std::unordered_map<std::string, std::string>& option_map) {
  RegisteredEpDeviceUniquePtr nv_tensorrt_rtx_ep;
  /// Since this test runs after other tests that use registration interface this test has to use it as well
  /// windows as otherwise the kernel registry inside the EP will not be populated. The legacy APis ony call the initialize once.
  Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, nv_tensorrt_rtx_ep);
  auto ep_devices = ort_env->GetEpDevices();
  Ort::ConstEpDevice selected_device;
  for (auto& device : ep_devices) {
    if (!std::strcmp(device.EpName(), kNvTensorRTRTXExecutionProvider)) {
      selected_device = device;
    }
  }
  session_options.AppendExecutionProvider_V2(*ort_env, {selected_device}, option_map);
  return nv_tensorrt_rtx_ep;
}

std::vector<char> readBinaryFile(const PathString& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + PathToUTF8String(filename));
  }

  file.seekg(0, std::ios::end);
  std::streamsize filesize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(filesize);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), filesize)) {
    throw std::runtime_error("Could not read file: " + PathToUTF8String(filename));
  }

  return buffer;
}

struct CompileParam {
  bool embed_mode;
  bool bytestream_io;
  bool external_initialzier_for_parser = false;
  const std::string to_string() const {
    return "embed_mode_" + std::to_string(embed_mode) + "_bytestream_io_" + std::to_string(bytestream_io) + "_ext_init_" + std::to_string(external_initialzier_for_parser);
    ;
  }
};
class CompileApiTest
    : public testing::TestWithParam<CompileParam> {
 public:
  const CompileParam& GetCompileParam() const {
    return GetParam();
  }
};

void SmallModelTest(CompileParam test_param, bool fully_supported_model) {
  std::string test_name = test_param.to_string();
  if (!fully_supported_model)
    test_name += "_fast_gelu";
  PathString model_name = path_utils::MakePathString("nv_execution_provider_compile_" + test_name + ".onnx");
  PathString model_name_ctx = path_utils::MakePathString("nv_execution_provider_compile_" + test_name + "_ctx.onnx");
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims, !fully_supported_model);

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map{
      {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer, std::to_string(test_param.external_initialzier_for_parser)}};
  auto ep = AppendTrtEtxEP(session_options, option_map);

  Ort::ModelCompilationOptions model_compile_options(*ort_env, session_options);
  model_compile_options.SetEpContextEmbedMode(test_param.embed_mode);

  void* output_context = nullptr;
  size_t output_context_size = 0;
  std::vector<char> input_onnx;
  if (test_param.bytestream_io) {
    input_onnx = readBinaryFile(model_name);
    model_compile_options.SetInputModelFromBuffer(input_onnx.data(), input_onnx.size());
    model_compile_options.SetOutputModelBuffer(Ort::AllocatorWithDefaultOptions(), &output_context, &output_context_size);
  } else {
    model_compile_options.SetInputModelPath(model_name.c_str());
    model_compile_options.SetOutputModelPath(model_name_ctx.c_str());
  }
  // AOT time
  ASSERT_TRUE(Ort::CompileModel(*ort_env, model_compile_options).IsOK());

  // JIT time
  Ort::Session session_object{nullptr};
  if (test_param.bytestream_io) {
    session_object = Ort::Session(*ort_env, output_context, output_context_size, session_options);
  } else {
    session_object = Ort::Session(*ort_env, model_name_ctx.c_str(), session_options);
  }
  auto io_binding = generate_io_binding(session_object);
  Ort::RunOptions run_options;
  session_object.Run(run_options, io_binding);
}

TEST_P(CompileApiTest, SmallModel) {
  const auto& test_param = GetCompileParam();
  SmallModelTest(test_param, true);
}

TEST_P(CompileApiTest, SmallSplitModel) {
  const auto& test_param = GetCompileParam();
  SmallModelTest(test_param, false);
}

TEST_P(CompileApiTest, LargeModel) {
  const auto& test_param = GetCompileParam();
  // with embed mode == 1 the resulting file will be over the 2GB proto limit
  if (test_param.embed_mode == 1) {
    GTEST_SKIP();
  }
  std::string test_name = test_param.to_string();
  PathString model_name = path_utils::MakePathString("nv_execution_provider_compile_large_" + test_name + ".onnx");
  PathString external_data_name = path_utils::MakePathString("nv_execution_provider_compile_large_" + test_name + ".onnx_data");
  PathString model_name_ctx = path_utils::MakePathString("nv_execution_provider_compile_large_" + test_name + "_ctx.onnx");
  PathString model_name_ctx_data = path_utils::MakePathString("nv_execution_provider_compile_large_" + test_name + "_ctx.onnx_data");
  clearFileIfExists(model_name_ctx);
  clearFileIfExists(model_name_ctx_data);
  // This accelerates test iterations if the large model was already generated
  if (!std::filesystem::exists(model_name) || !std::filesystem::exists(external_data_name)) {
    CreateLargeLLMModel(model_name, external_data_name);
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map{
      {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer,
       std::to_string(test_param.bytestream_io || test_param.external_initialzier_for_parser)}};
  auto ep = AppendTrtEtxEP(session_options, option_map);

  Ort::ModelCompilationOptions model_compile_options(*ort_env, session_options);
  model_compile_options.SetEpContextEmbedMode(test_param.embed_mode);

  void* output_context = nullptr;
  size_t output_context_size = 0;
  std::vector<char> input_onnx, input_data;
  std::vector<PathString> file_names;
  std::vector<char*> file_buffers;
  std::vector<size_t> lengths;
  if (test_param.bytestream_io) {
    input_onnx = readBinaryFile(model_name);
    input_data = readBinaryFile(external_data_name);
    file_names = {external_data_name};
    file_buffers = {input_data.data()};
    lengths = {input_data.size()};
    session_options.AddExternalInitializersFromFilesInMemory(file_names, file_buffers, lengths);

    model_compile_options.SetInputModelFromBuffer(input_onnx.data(), input_onnx.size());
    model_compile_options.SetOutputModelBuffer(Ort::AllocatorWithDefaultOptions(), &output_context, &output_context_size);
  } else {
    model_compile_options.SetInputModelPath(model_name.c_str());
    model_compile_options.SetOutputModelPath(model_name_ctx.c_str());
    model_compile_options.SetOutputModelExternalInitializersFile(model_name_ctx_data.c_str(), 1024);
  }

  // AOT time
  ASSERT_TRUE(Ort::CompileModel(*ort_env, model_compile_options).IsOK());

  // JIT time
  std::unique_ptr<Ort::Session> session;
  if (test_param.bytestream_io) {
    session = std::make_unique<Ort::Session>(*ort_env, output_context, output_context_size, session_options);
  } else {
    session = std::make_unique<Ort::Session>(*ort_env, model_name_ctx.c_str(), session_options);
  }

  auto io_binding = generate_io_binding(*session);
  Ort::RunOptions run_options;
  session->Run(run_options, io_binding);
}

INSTANTIATE_TEST_SUITE_P(
    NvExecutionProviderTest, CompileApiTest,
    ::testing::Values(
        CompileParam{true, false},
        CompileParam{false, false},
        CompileParam{true, true},
        CompileParam{false, true},
        // test with external initializers for parser
        CompileParam{true, true, true},
        CompileParam{true, false, true}),
    [](const testing::TestParamInfo<CompileApiTest::ParamType>& info) {
      return info.param.to_string();
    });

/*
 * Helper to create a synthetic EPContext ONNX model with a specific "source" attribute.
 * Uses raw ONNX protobuf to bypass schema validation (EPContext is a contrib op).
 */
void CreateSyntheticEPContextModel(const PathString& model_path,
                                   const std::string& source_attr,
                                   bool include_source_attr = true) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(11);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("EPContextSourceTest");

  // Input
  auto* input = graph->add_input();
  input->set_name("input");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_value(1);
  input_type->mutable_shape()->add_dim()->set_dim_value(3);

  // Output
  auto* output = graph->add_output();
  output->set_name("output");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_type->mutable_shape()->add_dim()->set_dim_value(1);
  output_type->mutable_shape()->add_dim()->set_dim_value(3);

  // EPContext node
  auto* node = graph->add_node();
  node->set_op_type("EPContext");
  node->set_domain("com.microsoft");
  node->set_name("ep_context_node");
  node->add_input("input");
  node->add_output("output");

  // embed_mode attribute
  auto* attr_embed = node->add_attribute();
  attr_embed->set_name("embed_mode");
  attr_embed->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_embed->set_i(1);

  // ep_cache_context attribute (dummy data)
  auto* attr_cache = node->add_attribute();
  attr_cache->set_name("ep_cache_context");
  attr_cache->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  attr_cache->set_s("dummy_context_data");

  // source attribute (conditionally added)
  if (include_source_attr) {
    auto* attr_source = node->add_attribute();
    attr_source->set_name("source");
    attr_source->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    attr_source->set_s(source_attr);
  }

  // Save to file
  std::ofstream ofs(model_path, std::ios::binary);
  ASSERT_TRUE(ofs.is_open());
  ASSERT_TRUE(model.SerializeToOstream(&ofs));
}

/*
 * Test: NvTensorRTRTX EP should NOT claim an EPContext node whose "source"
 * attribute belongs to a different EP (e.g., OpenVINO).
 *
 * Expected: Session initialization fails because no EP claims the node.
 */
TEST(NvExecutionProviderTest, EPContextNode_ForeignSourceSkipped) {
  PathString model_path = path_utils::MakePathString("ep_context_foreign_source_nv.onnx");
  CreateSyntheticEPContextModel(model_path, "OpenVINOExecutionProvider");

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map;
  auto ep = AppendTrtEtxEP(session_options, option_map);

  // Loading a model with a foreign-source EPContext node should fail during
  // session creation because the NvTensorRTRTX EP correctly skips the node
  // and no other EP can handle it.
  try {
    Ort::Session session(*ort_env, model_path.c_str(), session_options);
    FAIL() << "Expected session creation to fail for EPContext node with foreign source";
  } catch (const Ort::Exception& e) {
    std::string error_msg = e.what();
    EXPECT_TRUE(error_msg.find("EPContext") != std::string::npos)
        << "Error should mention EPContext. Actual: " << error_msg;
  }

  // Clean up
  std::filesystem::remove(model_path);
}

/*
 * Test: NvTensorRTRTX EP should NOT claim an EPContext node whose "source"
 * attribute is set to the classic TensorRT EP name.
 */
TEST(NvExecutionProviderTest, EPContextNode_ClassicTrtSourceSkipped) {
  PathString model_path = path_utils::MakePathString("ep_context_classic_trt_source_nv.onnx");
  CreateSyntheticEPContextModel(model_path, "TensorrtExecutionProvider");

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map;
  auto ep = AppendTrtEtxEP(session_options, option_map);

  try {
    Ort::Session session(*ort_env, model_path.c_str(), session_options);
    FAIL() << "Expected session creation to fail for EPContext node with classic TRT source";
  } catch (const Ort::Exception& e) {
    std::string error_msg = e.what();
    EXPECT_TRUE(error_msg.find("EPContext") != std::string::npos)
        << "Error should mention EPContext. Actual: " << error_msg;
  }

  // Clean up
  std::filesystem::remove(model_path);
}

/*
 * Test: NvTensorRTRTX EP should still claim an EPContext node that has NO
 * "source" attribute (backward compatibility with legacy context models).
 *
 * Expected: The EP claims the node. It may fail later during engine
 * deserialization (since context data is synthetic), but the error must NOT
 * be "is not compatible with any execution provider", which would indicate
 * the node was not claimed at all.
 */
TEST(NvExecutionProviderTest, EPContextNode_NoSourceAttribute_BackwardCompat) {
  PathString model_path = path_utils::MakePathString("ep_context_no_source_nv.onnx");
  CreateSyntheticEPContextModel(model_path, "", /*include_source_attr=*/false);

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map;
  auto ep = AppendTrtEtxEP(session_options, option_map);

  try {
    Ort::Session session(*ort_env, model_path.c_str(), session_options);
    // If session creation succeeds, backward compatibility is working.
  } catch (const Ort::Exception& e) {
    std::string error_msg = e.what();
    // The node should have been claimed by the EP. Any failure should be
    // EP-internal (e.g., bad engine data), NOT the "not compatible" error
    // that indicates no EP claimed the node.
    EXPECT_TRUE(error_msg.find("is not compatible with any execution provider") == std::string::npos)
        << "Legacy EPContext node without source should still be claimed by EP. Error: " << error_msg;
  }

  // Clean up
  std::filesystem::remove(model_path);
}

}  // namespace test
}  // namespace onnxruntime
