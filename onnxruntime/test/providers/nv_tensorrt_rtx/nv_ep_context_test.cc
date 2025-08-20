// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/common/path_utils.h"
#include "test/framework/test_utils.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <fstream>

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

RegisteredEpDeviceUniquePtr AppendTrtEtxEP(Ort::SessionOptions& session_options, std::unordered_map<std::string, std::string>& option_map) {
  RegisteredEpDeviceUniquePtr nv_tensorrt_rtx_ep;
#ifdef _WIN32
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
#else
  session_options.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, option_map);
#endif
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

}  // namespace test
}  // namespace onnxruntime
