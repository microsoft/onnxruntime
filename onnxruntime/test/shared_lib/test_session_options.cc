// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gmock/gmock.h"
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
#include "absl/flags/reflection.h"
#endif

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/resource.h>
#endif
#if (defined(_WIN32) || defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)) && \
    !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
#if defined(__linux__)
#include "absl/base/config.h"
#endif
#include "core/graph/onnx_protobuf.h"
#include "test/util/include/scoped_env_vars.h"
#endif

using namespace onnxruntime;

TEST(CApiTest, session_options_graph_optimization_level) {
  // Test set optimization level succeeds when valid level is provided.
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
}

TEST(CApiTest, session_options_deterministic_compute) {
  // Manual validation currently. Check that SetDeterministicCompute in abi_session_options.cc is hit.
  Ort::SessionOptions options;
  options.SetDeterministicCompute(true);
}

TEST(CApiTest, session_options_get_mem_pattern_enabled) {
  Ort::SessionOptions options;

  // Memory pattern is enabled by default
  ASSERT_TRUE(options.GetMemPatternEnabled());

  // Disable and verify
  options.DisableMemPattern();
  ASSERT_FALSE(options.GetMemPatternEnabled());

  // Re-enable and verify
  options.EnableMemPattern();
  ASSERT_TRUE(options.GetMemPatternEnabled());
}

TEST(CApiTest, session_options_get_execution_mode) {
  Ort::SessionOptions options;

  // Default is sequential
  ASSERT_EQ(options.GetExecutionMode(), ORT_SEQUENTIAL);

  // Set to parallel and verify
  options.SetExecutionMode(ORT_PARALLEL);
  ASSERT_EQ(options.GetExecutionMode(), ORT_PARALLEL);

  // Set back to sequential and verify
  options.SetExecutionMode(ORT_SEQUENTIAL);
  ASSERT_EQ(options.GetExecutionMode(), ORT_SEQUENTIAL);
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD) && !defined(ORT_NO_EXCEPTIONS)

TEST(CApiTest, session_options_oversized_affinity_string) {
  Ort::SessionOptions options;
  std::string long_affinity_str(ConfigOptions::kMaxValueLength + 1, '0');
  try {
    options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, long_affinity_str.c_str());
    ASSERT_TRUE(false) << "Creation of config should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Config value is longer than maximum length: "));
  }
}

#endif

#if defined(USE_OPENVINO_PROVIDER_INTERFACE)
// Test that loading OpenVINO EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_openvino) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kOpenVINOExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_OPENVINO_PROVIDER_INTERFACE)

#if defined(USE_CUDA_PROVIDER_INTERFACE)
// Test that loading CUDA EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_cuda) {
  Ort::SessionOptions session_options;

  Ort::CUDAProviderOptions cuda_options;
  try {
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    FAIL() << "Appending CUDA options have thrown exception";
  } catch (const Ort::Exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Failed to load"));
  }
}
#endif  // defined(USE_CUDA_PROVIDER_INTERFACE)

#if !defined(ORT_MINIMAL_BUILD)
// Test that appending the deprecated provider-bridge NV TensorRT RTX EP by name reports the plugin EP flow.
TEST(CApiTest, session_options_append_removed_nv_tensorrt_rtx_ep) {
  const OrtApi& api = Ort::GetApi();

  for (const char* provider_name : {"NvTensorRtRtx", kNvTensorRTRTXExecutionProvider}) {
    Ort::SessionOptions session_options;
    Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options, provider_name,
                                                                               nullptr, nullptr, 0)};
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Use the NVIDIA TensorRT RTX plugin EP"));
  }
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if defined(USE_TENSORRT_PROVIDER_INTERFACE)
// Test that loading TensorRT EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_add_tensorrt) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  OrtTensorRTProviderOptionsV2* trt_options = nullptr;
  Ort::Status status1 = Ort::Status{api.CreateTensorRTProviderOptions(&trt_options)};
  ASSERT_TRUE(status1.IsOK());

  Ort::Status status2 = Ort::Status{api.SessionOptionsAppendExecutionProvider_TensorRT_V2(session_options,
                                                                                          trt_options)};
  ASSERT_FALSE(status2.IsOK());
  EXPECT_EQ(status2.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status2.GetErrorMessage(), testing::HasSubstr("Failed to load"));

  api.ReleaseTensorRTProviderOptions(trt_options);
}
#endif  // defined(USE_TENSORRT_PROVIDER_INTERFACE)

#if defined(USE_VITISAI_PROVIDER_INTERFACE)
// Test that loading VitisAI EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_vitisai) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kVitisAIExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_VITISAI_PROVIDER_INTERFACE)

#if defined(USE_QNN_PROVIDER_INTERFACE)
// Test that loading QNN EP when only the interface is built (but not the full EP) fails.
TEST(CApiTest, session_options_provider_interface_fail_qnn) {
  const OrtApi& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  Ort::Status status = Ort::Status{api.SessionOptionsAppendExecutionProvider(session_options,
                                                                             kQnnExecutionProvider,
                                                                             nullptr, nullptr, 0)};
  ASSERT_TRUE(!status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_THAT(status.GetErrorMessage(), testing::HasSubstr("Failed to load"));
}
#endif  // defined(USE_QNN_PROVIDER_INTERFACE)

#if (defined(_WIN32) || defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)) && !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
namespace {
OrtFileHandle NativeFileHandle(FILE* file) {
#ifdef _WIN32
  return reinterpret_cast<OrtFileHandle>(_get_osfhandle(_fileno(file)));
#else
  return fileno(file);
#endif
}

#ifdef _WIN32
constexpr OrtFileHandle kInvalidFileHandle = nullptr;
#else
constexpr OrtFileHandle kInvalidFileHandle = -1;
#endif

struct ScopedInitializerDirectory {
  ScopedInitializerDirectory()
      : path{std::filesystem::temp_directory_path() /
             ("ort-file-source-" +
              std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))} {
    ORT_ENFORCE(std::filesystem::create_directory(path), "Cannot create initializer test directory");
  }
  ~ScopedInitializerDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
    EXPECT_FALSE(error) << error.message();
  }
  const std::filesystem::path path;
};

std::unique_ptr<FILE, decltype(&fclose)> OpenReadOnlyInitializer(const std::filesystem::path& path) {
  FILE* file = nullptr;
#ifdef _WIN32
  const auto error = _wfopen_s(&file, path.c_str(), L"rb");
  ORT_ENFORCE(error == 0, "Cannot open initializer test file");
#else
  file = fopen(path.c_str(), "rb");
#endif
  ORT_ENFORCE(file != nullptr, "Cannot open initializer test file");
  return {file, fclose};
}

Ort::Status TryCreateSessionFromArray(const OrtApi* api, const Ort::Env& env,
                                      const void* model_data, size_t model_data_length,
                                      const Ort::SessionOptions& options) {
  OrtSession* raw_session = nullptr;
  OrtStatus* raw_status = api->CreateSessionFromArray(env, model_data, model_data_length, options, &raw_session);
  if (raw_session != nullptr) {
    api->ReleaseSession(raw_session);
  }
  return Ort::Status{raw_status};
}

ONNX_NAMESPACE::ModelProto MakeFileSourceTestModel(int64_t length = 1) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.add_opset_import()->set_version(18);
  auto* graph = model.mutable_graph();
  graph->set_name("file-source-test");
  auto* node = graph->add_node();
  node->set_op_type("Identity");
  node->add_input("X");
  node->add_output("Y");
  const auto set_type = [length](ONNX_NAMESPACE::ValueInfoProto* value, const char* name) {
    value->set_name(name);
    auto* tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor->mutable_shape()->add_dim()->set_dim_value(length);
  };
  set_type(graph->add_input(), "X");
  set_type(graph->add_output(), "Y");
  return model;
}

void SetExternalFile(ONNX_NAMESPACE::TensorProto& tensor, const char* path, int64_t offset = 0) {
  tensor.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* location = tensor.add_external_data();
  location->set_key("location");
  location->set_value(path);
  auto* start = tensor.add_external_data();
  start->set_key("offset");
  start->set_value(std::to_string(offset));
}

ONNX_NAMESPACE::ModelProto MakeExternalInitializerModel(int64_t length, int64_t offset) {
  auto model = MakeFileSourceTestModel(length);
  auto* graph = model.mutable_graph();
  graph->clear_input();
  auto* tensor = graph->add_initializer();
  tensor->set_name("X");
  tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor->add_dims(length);
  SetExternalFile(*tensor, "weights.bin", offset);
  return model;
}

ONNX_NAMESPACE::ModelProto MakeNestedFileSourceModel(int64_t length, int depth,
                                                     const std::string& location) {
  auto model = MakeExternalInitializerModel(length, 0);
  auto* graph = model.mutable_graph();
  graph->mutable_initializer(0)->mutable_external_data(0)->set_value(location);
  if (depth == 0) {
    return model;
  }
  graph->clear_initializer();
  auto* condition = graph->add_input();
  condition->set_name("condition");
  auto* type = condition->mutable_type()->mutable_tensor_type();
  type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  type->mutable_shape();
  auto* node = graph->mutable_node(0);
  node->set_op_type("If");
  node->clear_input();
  node->add_input("condition");
  auto branch = MakeNestedFileSourceModel(length, depth - 1, location);
  branch.mutable_graph()->clear_input();
  for (const char* name : {"then_branch", "else_branch"}) {
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    attribute->mutable_g()->CopyFrom(branch.graph());
  }
  return model;
}

ONNX_NAMESPACE::ModelProto MakeFileSourceLoopModel(int64_t length, const std::string& location) {
  auto model = MakeFileSourceTestModel(length);
  auto* graph = model.mutable_graph();
  graph->mutable_input(0)->set_name("state");
  auto* trips = graph->add_initializer();
  trips->set_name("trips");
  trips->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  trips->add_int64_data(2);
  auto* condition = graph->add_initializer();
  condition->set_name("condition");
  condition->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  condition->add_int32_data(1);
  auto* loop = graph->mutable_node(0);
  loop->set_op_type("Loop");
  loop->clear_input();
  for (const char* name : {"trips", "condition", "state"}) {
    loop->add_input(name);
  }
  auto body_model = MakeExternalInitializerModel(length, 0);
  auto* body = body_model.mutable_graph();
  body->set_name("file-source-loop-body");
  body->mutable_initializer(0)->mutable_external_data(0)->set_value(location);
  for (const auto& entry : {std::pair{"iteration", ONNX_NAMESPACE::TensorProto_DataType_INT64},
                            std::pair{"condition_in", ONNX_NAMESPACE::TensorProto_DataType_BOOL}}) {
    auto* input = body->add_input();
    input->set_name(entry.first);
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(entry.second);
    type->mutable_shape();
  }
  auto* state = body->add_input();
  state->CopyFrom(graph->input(0));
  state->set_name("state_in");
  auto* add = body->mutable_node(0);
  add->set_op_type("Add");
  add->clear_input();
  add->add_input("state_in");
  add->add_input("X");
  add->set_output(0, "state_out");
  body->mutable_output(0)->set_name("state_out");
  auto* continue_loop = body->add_node();
  continue_loop->set_op_type("Identity");
  continue_loop->add_input("condition_in");
  continue_loop->add_output("condition_out");
  auto* condition_out = body->add_output();
  condition_out->CopyFrom(body->input(1));
  condition_out->set_name("condition_out");
  body->mutable_output()->SwapElements(0, 1);
  auto* attribute = loop->add_attribute();
  attribute->set_name("body");
  attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  attribute->mutable_g()->CopyFrom(*body);
  return model;
}

#if defined(__linux__) && GTEST_HAS_DEATH_TEST && !defined(ABSL_HAVE_ADDRESS_SANITIZER) && \
    !defined(ABSL_HAVE_MEMORY_SANITIZER) && !defined(ABSL_HAVE_THREAD_SANITIZER) &&        \
    !defined(ABSL_HAVE_HWADDRESS_SANITIZER)
// Shadow/quarantine memory is not part of the unsanitized payload-copy budget.
int CheckSubgraphLoadingMemory(int depth) {
  constexpr size_t kWeightBytes = 32 * 1024 * 1024;
  constexpr long kMaximumGrowthKiB = kWeightBytes / 2 / 1024;
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  {
    std::ofstream output(path, std::ios::binary);
    ORT_ENFORCE(output.good(), "Cannot create initializer test file");
  }
  std::filesystem::resize_file(path, kWeightBytes);
  auto file = OpenReadOnlyInitializer(path);
  const auto handle = NativeFileHandle(file.get());
  const auto location = directory.path.filename().string() + ".bin";
  const char* name = location.c_str();
  auto model = MakeNestedFileSourceModel(kWeightBytes / sizeof(float), depth, location);
  const auto bytes = model.SerializeAsString();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "bounded-subgraph-checker"};
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetExternalInitializerFileHandles(&name, &handle, 1);
  file.reset();
  struct rusage before{}, after{};
  if (getrusage(RUSAGE_SELF, &before) != 0) {
    std::fprintf(stderr, "Cannot measure peak resident memory before model loading\n");
    return 1;
  }
  Ort::Session session{env, bytes.data(), bytes.size(), options};
  if (getrusage(RUSAGE_SELF, &after) != 0) {
    std::fprintf(stderr, "Cannot measure peak resident memory after model loading\n");
    return 1;
  }
  const auto growth = after.ru_maxrss - before.ru_maxrss;
  std::printf("depth=%d, peak growth=%ld KiB, budget=%ld KiB\n",
              depth, growth, kMaximumGrowthKiB);
  std::fflush(stdout);
  return growth < kMaximumGrowthKiB ? 0 : 1;
}
#endif

void SetSparseTestData(ONNX_NAMESPACE::SparseTensorProto& sparse, bool external_indices) {
  sparse.add_dims(1);
  auto* values = sparse.mutable_values();
  values->set_name("X");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(1);
  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(1);
  if (external_indices) {
    values->add_float_data(1.0f);
    SetExternalFile(*indices, "testdata/Pads.bin");
  } else {
    indices->add_int64_data(0);
    SetExternalFile(*values, "testdata/Pads.bin");
  }
}
}  // namespace
#endif

class FileSourcesSessionOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ASSERT_NE(api_, nullptr);
  }

  const OrtApi* api_ = nullptr;
};

#if (defined(_WIN32) || defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)) && !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
TEST_F(FileSourcesSessionOptionsTest, LegacyOrtLoopLoadsWithoutFileSources) {
  const auto path = ORT_TSTR("testdata/ort_github_issue_4031.onnx.ort");
  std::ifstream input("testdata/ort_github_issue_4031.onnx.ort", std::ios::binary);
  ASSERT_TRUE(input);
  const std::vector<char> bytes{std::istreambuf_iterator<char>{input}, {}};
  ASSERT_FALSE(bytes.empty());
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "legacy-ort-loop"};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t shape[]{1};
  float value = 123.0f;
  auto tensor = Ort::Value::CreateTensor<float>(memory, &value, 1, shape, 1);
  const char* inputs[]{"state_var_in"};
  const char* outputs[]{"state_var_out"};
  for (int mode : {0, 1, 2, 3, 4}) {
    SCOPED_TRACE(mode);
    Ort::SessionOptions options;
    if (mode == 2 || mode == 3) {
      options.AddConfigEntry("session.use_ort_model_bytes_directly", "1");
    }
    if (mode == 3) {
      options.AddConfigEntry("session.use_ort_model_bytes_for_initializers", "1");
    }
    if (mode == 4) {
      options.AddConfigEntry("session.use_memory_mapped_ort_model", "1");
    }
    auto session = mode > 0 && mode < 4
                       ? Ort::Session{env, bytes.data(), bytes.size(), options}
                       : Ort::Session{env, path, options};
    auto actual = session.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
    ASSERT_EQ(actual.size(), 1u);
    ASSERT_EQ(actual[0].GetTensorTypeAndShapeInfo().GetShape(), std::vector<int64_t>{1});
    EXPECT_FLOAT_EQ(actual[0].GetTensorData<float>()[0], 125.0f);
  }
}

TEST_F(FileSourcesSessionOptionsTest, LegacyOrtLoopRoundTripWithoutFileSources) {
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "loop.ort";
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "legacy-ort-loop-round-trip"};
  {
    Ort::SessionOptions options;
    options.AddConfigEntry("session.save_model_format", "ORT");
    options.SetOptimizedModelFilePath(path.c_str());
    Ort::Session original{env, ORT_TSTR("testdata/ort_github_issue_4031.onnx"), options};
  }
  Ort::Session restored{env, path.c_str(), Ort::SessionOptions{}};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t shape[]{1};
  float value = 123.0f;
  auto tensor = Ort::Value::CreateTensor<float>(memory, &value, 1, shape, 1);
  const char* inputs[]{"state_var_in"};
  const char* outputs[]{"state_var_out"};
  auto actual = restored.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
  ASSERT_EQ(actual.size(), 1u);
  ASSERT_EQ(actual[0].GetTensorTypeAndShapeInfo().GetShape(), std::vector<int64_t>{1});
  EXPECT_FLOAT_EQ(actual[0].GetTensorData<float>()[0], 125.0f);
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesValidateArgumentsAndAccess) {
  Ort::SessionOptions options;
  EXPECT_FALSE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(nullptr, nullptr, nullptr, 0)).IsOK());
  EXPECT_FALSE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 1)).IsOK());
#ifdef _WIN32
  const auto temporary_path =
      std::filesystem::temp_directory_path() /
      ("ort-file-source-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  const HANDLE handle = CreateFileW(
      temporary_path.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
      FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
      FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE) << GetLastError();
  std::unique_ptr<void, decltype(&CloseHandle)> file(handle, CloseHandle);
  const OrtFileHandle descriptor = file.get();
#else
  std::unique_ptr<FILE, decltype(&fclose)> file(tmpfile(), fclose);
  ASSERT_NE(file, nullptr);
  const auto descriptor = NativeFileHandle(file.get());
#endif
  const char* name = "weights.data";
#ifdef _WIN32
  // Windows duplicates only read rights, even when the original also permits writing.
  EXPECT_TRUE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &descriptor, 1)).IsOK());
#else
  EXPECT_FALSE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &descriptor, 1)).IsOK());
#endif
  const auto invalid = kInvalidFileHandle;
  EXPECT_FALSE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &invalid, 1)).IsOK());
  for (const char* bad : {"../weights.data", "/weights.data", ".", "a/b"}) {
    EXPECT_FALSE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(options, &bad, &invalid, 1)).IsOK());
  }
#ifdef O_PATH
  const int path_only = open("testdata/Pads.bin", O_PATH | O_CLOEXEC);
  ASSERT_GE(path_only, 0);
  EXPECT_FALSE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &path_only, 1)).IsOK());
  EXPECT_EQ(close(path_only), 0);
#endif
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesCloneAndRejectBufferMixing) {
  Ort::SessionOptions options;
  ASSERT_TRUE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 0)).IsOK());
  auto clone = options.Clone();
  char data[8]{};
  char* buffer = data;
  const ORTCHAR_T* name = ORT_TSTR("weights.data");
  const size_t size = sizeof(data);
  EXPECT_FALSE(Ort::Status(api_->AddExternalInitializersFromFilesInMemory(
                               clone, &name, &buffer, &size, 1))
                   .IsOK());
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesInferAfterHandleAndOptionsRelease) {
  std::ifstream input("testdata/model_with_external_initializers.onnx", std::ios::binary);
  ASSERT_TRUE(input);
  const std::vector<char> graph{std::istreambuf_iterator<char>{input}, {}};
  FILE* raw_weights = nullptr;
#ifdef _WIN32
  ASSERT_EQ(_wfopen_s(&raw_weights, ORT_TSTR("testdata/Pads.bin"), ORT_TSTR("rb")), 0);
#else
  raw_weights = fopen("testdata/Pads.bin", "rb");
#endif
  std::unique_ptr<FILE, decltype(&fclose)> weights(raw_weights, fclose);
  ASSERT_NE(weights, nullptr);
  const char* name = "Pads.bin";
  const auto descriptor = NativeFileHandle(weights.get());
  Ort::SessionOptions options;
  options.SetExternalInitializerFileHandles(&name, &descriptor, 1);
  auto clone = options.Clone();
  weights.reset();
  options = Ort::SessionOptions{nullptr};
  const auto invalid = kInvalidFileHandle;
  EXPECT_FALSE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(clone, &name, &invalid, 1)).IsOK());

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "file-sources"};
  Ort::Session session{env, graph.data(), graph.size(), clone};
  clone = Ort::SessionOptions{nullptr};
  Ort::Session reference{env, ORT_TSTR("testdata/model_with_external_initializers.onnx"),
                         Ort::SessionOptions{}};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t shape[]{1, 2};
  float values[]{1.0f, 2.0f};
  auto tensor = Ort::Value::CreateTensor<float>(memory, values, 2, shape, 2);
  const char* inputs[]{"X"};
  const char* outputs[]{"Y"};
  auto expected = reference.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
  auto actual = session.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
  ASSERT_EQ(actual.size(), 1u);
  const auto expected_info = expected[0].GetTensorTypeAndShapeInfo();
  const auto actual_info = actual[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(actual_info.GetShape(), expected_info.GetShape());
  ASSERT_EQ(actual_info.GetElementCount(), expected_info.GetElementCount());
  const auto* expected_data = expected[0].GetTensorData<float>();
  const auto* actual_data = actual[0].GetTensorData<float>();
  for (size_t i = 0; i < actual_info.GetElementCount(); ++i) {
    EXPECT_FLOAT_EQ(actual_data[i], expected_data[i]);
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesNeverFallBackToModelPaths) {
  std::ifstream input("testdata/model_with_external_initializers.onnx", std::ios::binary);
  ASSERT_TRUE(input);
  const std::vector<char> graph{std::istreambuf_iterator<char>{input}, {}};
  ASSERT_FALSE(graph.empty());
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "file-sources"};
  Ort::SessionOptions options;
  ASSERT_TRUE(Ort::Status(api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 0)).IsOK());
  const auto status = TryCreateSessionFromArray(api_, env, graph.data(), graph.size(), options);
  ASSERT_FALSE(status.IsOK()) << "Missing registered sources must not fall back to disk";
  EXPECT_NE(status.GetErrorMessage().find("No registered external initializer file source"),
            std::string::npos);
  EXPECT_NE(status.GetErrorMessage().find("Pads.bin"), std::string::npos);
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRejectExternalSparseDataBeforeGraphConstruction) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "sparse-file-sources"};
  for (bool external_indices : {false, true}) {
    for (int location : {0, 1, 2, 3}) {
      SCOPED_TRACE(::testing::Message() << "indices=" << external_indices << ", location=" << location);
      auto model = MakeFileSourceTestModel();
      auto* graph = model.mutable_graph();
      if (location == 0) {
        graph->clear_input();
        SetSparseTestData(*graph->add_sparse_initializer(), external_indices);
      } else {
        auto* attribute = graph->mutable_node(0)->add_attribute();
        attribute->set_name("sparse_value");
        if (location == 1) {
          attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_SPARSE_TENSOR);
          SetSparseTestData(*attribute->mutable_sparse_tensor(), external_indices);
        } else if (location == 2) {
          attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_SPARSE_TENSORS);
          SetSparseTestData(*attribute->add_sparse_tensors(), external_indices);
        } else {
          attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
          auto* nested = attribute->mutable_g();
          nested->set_name("nested-sparse");
          SetSparseTestData(*nested->add_sparse_initializer(), external_indices);
        }
      }
      Ort::SessionOptions options;
      Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 0));
      const auto bytes = model.SerializeAsString();
      const auto status = TryCreateSessionFromArray(api_, env, bytes.data(), bytes.size(), options);
      ASSERT_FALSE(status.IsOK()) << "External sparse data must be rejected before graph construction";
      EXPECT_NE(status.GetErrorMessage().find("does not support external sparse tensors"), std::string::npos);
    }
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRejectExternalDataInModelLocalFunctions) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "function-file-sources"};
  for (int location : {0, 1, 2, 3}) {
    SCOPED_TRACE(location);
    auto model = MakeFileSourceTestModel();
    model.mutable_graph()->clear_input();
    auto* call = model.mutable_graph()->mutable_node(0);
    call->set_op_type("Local");
    call->set_domain("file_source_test");
    call->clear_input();
    auto* domain = model.add_opset_import();
    domain->set_domain("file_source_test");
    domain->set_version(1);
    auto* function = model.add_functions();
    function->set_name("Local");
    function->set_domain("file_source_test");
    function->add_opset_import()->set_version(18);
    function->add_output("Y");
    auto* node = function->add_node();
    node->set_op_type("Constant");
    node->add_output("Y");
    if (location < 2) {
      auto* attribute = location == 1 ? function->add_attribute_proto() : node->add_attribute();
      attribute->set_name("value");
      attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
      auto* tensor = attribute->mutable_t();
      tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      tensor->add_dims(1);
      SetExternalFile(*tensor, "testdata/Pads.bin");
      if (location == 1) {
        auto* reference = node->add_attribute();
        reference->set_name("value");
        reference->set_ref_attr_name("value");
        reference->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
      }
    } else {
      auto* condition = model.mutable_graph()->add_input();
      condition->set_name("condition");
      condition->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
      condition->mutable_type()->mutable_tensor_type()->mutable_shape();
      call->add_input("condition");
      function->add_input("condition");
      node->set_op_type("If");
      node->add_input("condition");
      for (const char* branch : {"then_branch", "else_branch"}) {
        auto* attribute = node->add_attribute();
        attribute->set_name(branch);
        attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
        attribute->mutable_g()->CopyFrom(MakeExternalInitializerModel(1, 0).graph());
        attribute->mutable_g()->mutable_initializer(0)->mutable_external_data(0)->set_value("testdata/Pads.bin");
        if (location == 3) {
          attribute->mutable_g()->clear_initializer();
          SetSparseTestData(*attribute->mutable_g()->add_sparse_initializer(), true);
        }
      }
    }
    Ort::SessionOptions options;
    Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 0));
    const auto bytes = model.SerializeAsString();
    const auto status = TryCreateSessionFromArray(api_, env, bytes.data(), bytes.size(), options);
    ASSERT_FALSE(status.IsOK()) << "External function data must not reach pathname loading";
    const std::string message = status.GetErrorMessage();
    EXPECT_TRUE(message.find("does not support external tensor attributes") != std::string::npos ||
                message.find("does not support external data in model-local functions") != std::string::npos ||
                message.find("does not support external sparse tensors") != std::string::npos)
        << message;
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesValidateZeroLengthRanges) {
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  {
    std::ofstream output(path, std::ios::binary);
    output.put('x');
    ASSERT_TRUE(output.good());
  }
  auto file = OpenReadOnlyInitializer(path);
  const auto handle = NativeFileHandle(file.get());
  const char* name = "weights.bin";
  Ort::SessionOptions options;
  Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &handle, 1));
  file.reset();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "empty-file-sources"};
  const auto invalid = MakeExternalInitializerModel(0, 2).SerializeAsString();
  const auto invalid_status = TryCreateSessionFromArray(api_, env, invalid.data(), invalid.size(), options);
  ASSERT_FALSE(invalid_status.IsOK()) << "An empty initializer cannot begin beyond EOF";
  EXPECT_NE(invalid_status.GetErrorMessage().find("Invalid registered external initializer range"),
            std::string::npos);
  const auto valid = MakeExternalInitializerModel(0, 1).SerializeAsString();
  EXPECT_TRUE(TryCreateSessionFromArray(api_, env, valid.data(), valid.size(), options).IsOK());
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRejectHiddenDuplicateExternalInitializers) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "duplicate-file-sources"};
  for (bool external_first : {false, true}) {
    SCOPED_TRACE(external_first);
    auto model = MakeExternalInitializerModel(1, 0);
    auto* graph = model.mutable_graph();
    graph->mutable_initializer(0)->mutable_external_data(0)->set_value("testdata/Pads.bin");
    auto* embedded = graph->add_initializer();
    embedded->set_name("X");
    embedded->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    embedded->add_dims(1);
    embedded->add_float_data(7.0f);
    if (!external_first) {
      graph->mutable_initializer()->SwapElements(0, 1);
    }
    Ort::SessionOptions options;
    Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 0));
    const auto bytes = model.SerializeAsString();
    const auto status = TryCreateSessionFromArray(api_, env, bytes.data(), bytes.size(), options);
    ASSERT_FALSE(status.IsOK()) << "Duplicate names must not hide an external initializer from the registry";
    EXPECT_NE(status.GetErrorMessage().find("duplicate initializer names: X"), std::string::npos);
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRetainLargeAlignedAndUnalignedInitializers) {
  const std::vector<float> expected(64, 3.25f);
  ASSERT_GT(expected.size() * sizeof(float), 127u) << "Exercise ownership above TensorProto's inline limit";
  for (int64_t offset : {0, 1}) {
    SCOPED_TRACE(offset);
    ScopedInitializerDirectory directory;
    const auto path = directory.path / "weights.bin";
    {
      std::ofstream output(path, std::ios::binary);
      if (offset != 0) output.put('x');
      output.write(reinterpret_cast<const char*>(expected.data()),
                   static_cast<std::streamsize>(expected.size() * sizeof(float)));
      ASSERT_TRUE(output.good());
    }
    auto file = OpenReadOnlyInitializer(path);
    const auto handle = NativeFileHandle(file.get());
    const char* name = "weights.bin";
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &handle, 1));
    auto clone = options.Clone();
    file.reset();
    options = Ort::SessionOptions{nullptr};
    const auto graph = MakeExternalInitializerModel(static_cast<int64_t>(expected.size()), offset).SerializeAsString();
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "large-file-sources"};
    Ort::Session session{env, graph.data(), graph.size(), clone};
    clone = Ort::SessionOptions{nullptr};
    const char* outputs[]{"Y"};
    for (int iteration = 0; iteration < 2; ++iteration) {
      auto actual = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 1);
      ASSERT_EQ(actual.size(), 1u);
      const auto info = actual[0].GetTensorTypeAndShapeInfo();
      ASSERT_EQ(info.GetShape(), std::vector<int64_t>{64});
      ASSERT_EQ(info.GetElementCount(), expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[0].GetTensorData<float>()[i], expected[i]);
      }
    }
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesNormalizeBoolInitializers) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "bool-file-sources"};
  for (int64_t length : {0, 127, 128, 256}) {
    SCOPED_TRACE(length);
    ScopedInitializerDirectory directory;
    const auto path = directory.path / "weights.bin";
    const uint8_t pattern[]{0, 1, 2, 255};
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = pattern[i % 4];
    }
    {
      std::ofstream output(path, std::ios::binary);
      if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      }
      ASSERT_TRUE(output.good());
    }
    auto file = OpenReadOnlyInitializer(path);
    const auto handle = NativeFileHandle(file.get());
    const char* name = "weights.bin";
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &handle, 1));
    file.reset();
    auto model = MakeExternalInitializerModel(length, 0);
    auto* graph = model.mutable_graph();
    graph->mutable_initializer(0)->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
    graph->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_BOOL);
    auto* cast = graph->add_node();
    cast->set_op_type("Cast");
    cast->add_input("X");
    cast->add_output("Z");
    auto* to = cast->add_attribute();
    to->set_name("to");
    to->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    to->set_i(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    auto* cast_output = graph->add_output();
    cast_output->CopyFrom(graph->output(0));
    cast_output->set_name("Z");
    cast_output->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_INT32);
    const auto serialized = model.SerializeAsString();
    Ort::Session session{env, serialized.data(), serialized.size(), options};
    options = Ort::SessionOptions{nullptr};
    const char* outputs[]{"Y", "Z"};
    auto actual = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 2);
    ASSERT_EQ(actual.size(), 2u);
    const auto* raw = static_cast<const uint8_t*>(actual[0].GetTensorRawData());
    for (size_t i = 0; i < bytes.size(); ++i) {
      const auto expected = bytes[i] == 0 ? 0 : 1;
      ASSERT_EQ(raw[i], expected) << "BOOL byte " << i;
      ASSERT_EQ(actual[1].GetTensorData<int32_t>()[i], expected) << "Cast element " << i;
    }
    std::ifstream original(path, std::ios::binary);
    ASSERT_TRUE(original);
    const std::vector<uint8_t> unchanged{std::istreambuf_iterator<char>{original}, {}};
    EXPECT_EQ(unchanged, bytes) << "Normalization must not modify the source file";
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesInferSubgraphsWithoutWeightPaths) {
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  const std::vector<float> expected(64, 3.25f);
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(expected.data()),
                 static_cast<std::streamsize>(expected.size() * sizeof(float)));
    ASSERT_TRUE(output.good());
  }
  const auto missing_path = directory.path.filename().string() + ".bin";
  ASSERT_FALSE(std::filesystem::exists(missing_path));
  const char* name = missing_path.c_str();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "subgraph-file-sources"};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char* inputs[]{"condition"};
  const char* outputs[]{"Y"};
  for (int depth : {1, 3}) {
    SCOPED_TRACE(depth);
    auto file = OpenReadOnlyInitializer(path);
    const auto handle = NativeFileHandle(file.get());
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &handle, 1));
    file.reset();
    const auto serialized = MakeNestedFileSourceModel(64, depth, missing_path).SerializeAsString();
    Ort::Session session{env, serialized.data(), serialized.size(), options};
    options = Ort::SessionOptions{nullptr};
    for (bool value : {false, true}) {
      auto input = Ort::Value::CreateTensor<bool>(memory, &value, 1, nullptr, 0);
      auto actual = session.Run(Ort::RunOptions{}, inputs, &input, 1, outputs, 1);
      ASSERT_EQ(actual.size(), 1u);
      ASSERT_EQ(actual[0].GetTensorTypeAndShapeInfo().GetElementCount(), expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[0].GetTensorData<float>()[i], expected[i]);
      }
    }
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesCheckLoopAndNestedGraphStructure) {
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  const std::vector<float> weights(64, 3.25f);
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(weights.data()),
                 static_cast<std::streamsize>(weights.size() * sizeof(float)));
    ASSERT_TRUE(output.good());
  }
  const auto location = directory.path.filename().string() + ".bin";
  ASSERT_FALSE(std::filesystem::exists(location));
  const char* name = location.c_str();
  auto file = OpenReadOnlyInitializer(path);
  const auto handle = NativeFileHandle(file.get());
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &handle, 1));
  file.reset();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "loop-file-sources"};
  const auto serialized = MakeFileSourceLoopModel(64, location).SerializeAsString();
  Ort::Session session{env, serialized.data(), serialized.size(), options};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t shape[]{64};
  std::vector<float> values(64, 1.0f);
  auto input = Ort::Value::CreateTensor<float>(memory, values.data(), values.size(), shape, 1);
  const char* inputs[]{"state"};
  const char* outputs[]{"Y"};
  auto actual = session.Run(Ort::RunOptions{}, inputs, &input, 1, outputs, 1);
  ASSERT_EQ(actual.size(), 1u);
  ASSERT_EQ(actual[0].GetTensorTypeAndShapeInfo().GetElementCount(), weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[0].GetTensorData<float>()[i], 7.5f);
  }
  auto malformed = MakeNestedFileSourceModel(64, 1, location);
  malformed.mutable_graph()->mutable_node(0)->mutable_attribute(0)->mutable_g()->clear_name();
  const auto invalid = malformed.SerializeAsString();
  const auto invalid_status = TryCreateSessionFromArray(api_, env, invalid.data(), invalid.size(), options);
  ASSERT_FALSE(invalid_status.IsOK()) << "The metadata-only view must still reject malformed graph structure";
  EXPECT_NE(invalid_status.GetErrorMessage().find("Field 'name' of 'graph'"), std::string::npos);
}

#if defined(__linux__) && GTEST_HAS_DEATH_TEST && !defined(ABSL_HAVE_ADDRESS_SANITIZER) && \
    !defined(ABSL_HAVE_MEMORY_SANITIZER) && !defined(ABSL_HAVE_THREAD_SANITIZER) &&        \
    !defined(ABSL_HAVE_HWADDRESS_SANITIZER)
TEST_F(FileSourcesSessionOptionsTest, FileSourcesBoundNestedCheckerMemory) {
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
  auto* death_test_style_flag = absl::FindCommandLineFlag("gtest_death_test_style");
  ASSERT_NE(death_test_style_flag, nullptr);
  const std::string previous_style = death_test_style_flag->CurrentValue();
  std::string flag_error;
  ASSERT_TRUE(death_test_style_flag->ParseFrom("threadsafe", &flag_error)) << flag_error;
#else
  const auto previous_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#endif
  // A fresh process makes the high-water mark independent of earlier tests.
  for (int depth : {1, 3}) {
    SCOPED_TRACE(depth);
    EXPECT_EXIT(
        {
          const int result = CheckSubgraphLoadingMemory(depth);
          _exit(result);
        },
        ::testing::ExitedWithCode(0), "");
  }
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
  flag_error.clear();
  EXPECT_TRUE(death_test_style_flag->ParseFrom(previous_style, &flag_error)) << flag_error;
#else
  GTEST_FLAG_SET(death_test_style, previous_style);
#endif
}
#endif

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRejectExternalInitializerConstantCollisions) {
  auto model = MakeExternalInitializerModel(1, 0);
  auto* graph = model.mutable_graph();
  graph->mutable_initializer(0)->mutable_external_data(0)->set_value("testdata/Pads.bin");
  auto* constant = graph->add_node();
  constant->set_op_type("Constant");
  constant->add_output("X");
  auto* attribute = constant->add_attribute();
  attribute->set_name("value");
  attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* tensor = attribute->mutable_t();
  tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor->add_dims(1);
  tensor->add_float_data(7.0f);
  Ort::SessionOptions options;
  Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 0));
  const auto serialized = model.SerializeAsString();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "constant-file-sources"};
  const auto status = TryCreateSessionFromArray(api_, env, serialized.data(), serialized.size(), options);
  ASSERT_FALSE(status.IsOK()) << "Constant outputs must not hide external initializers from the registry";
  EXPECT_NE(status.GetErrorMessage().find("duplicate initializer names: X"), std::string::npos);
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesSurviveModelSuppliedSessionOptions) {
  const onnxruntime::test::ScopedEnvironmentVariables environment{{{"ORT_LOAD_CONFIG_FROM_MODEL", "1"}}};
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "model-config-file-sources"};
  auto model = MakeExternalInitializerModel(1, 0);
  auto* config = model.add_metadata_props();
  config->set_key("ort_config");
  config->set_value(R"({"session_options":{"intra_op_num_threads":1,"graph_optimization_level":0}})");
  model.mutable_graph()->mutable_initializer(0)->mutable_external_data(0)->set_value("testdata/Pads.bin");
  const auto missing = model.SerializeAsString();
  Ort::SessionOptions empty;
  Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(empty, nullptr, nullptr, 0));
  const auto missing_status = TryCreateSessionFromArray(api_, env, missing.data(), missing.size(), empty);
  ASSERT_FALSE(missing_status.IsOK()) << "Model-supplied options must not discard a strict empty registry";
  EXPECT_NE(missing_status.GetErrorMessage().find("No registered external initializer file source"), std::string::npos);
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  const float expected = 7.0f;
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&expected), sizeof(expected));
    ASSERT_TRUE(output.good());
  }
  auto file = OpenReadOnlyInitializer(path);
  const auto handle = NativeFileHandle(file.get());
  const char* name = "weights.bin";
  Ort::SessionOptions options;
  Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &name, &handle, 1));
  file.reset();
  model.mutable_graph()->mutable_initializer(0)->mutable_external_data(0)->set_value(name);
  const auto registered = model.SerializeAsString();
  Ort::Session session{env, registered.data(), registered.size(), options};
  options = Ort::SessionOptions{nullptr};
  const char* outputs[]{"Y"};
  auto actual = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 1);
  ASSERT_EQ(actual.size(), 1u);
  EXPECT_FLOAT_EQ(actual[0].GetTensorData<float>()[0], expected);
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRollbackFailedBatchAndReplaceNonEmptyRegistryAcrossClones) {
  ScopedInitializerDirectory directory;
  const auto write_weight_file = [&](const char* filename, float value) {
    const auto path = directory.path / filename;
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    ORT_ENFORCE(output.good(), "Cannot write test weight file");
    return path;
  };
  const auto path_a = write_weight_file("weights_a.bin", 10.0f);
  const auto path_provisional = write_weight_file("weights_provisional.bin", 20.0f);
  const auto path_b = write_weight_file("weights_b.bin", 30.0f);

#ifdef _WIN32
  const auto open_unshared_delete_file = [](const std::filesystem::path& path) {
    const HANDLE raw = CreateFileW(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    ORT_ENFORCE(raw != INVALID_HANDLE_VALUE, "Cannot open test weight file");
    return std::unique_ptr<void, decltype(&CloseHandle)>(raw, CloseHandle);
  };
  const auto has_open_handles = [](const std::filesystem::path& path) {
    const HANDLE probe = CreateFileW(
        path.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (probe == INVALID_HANDLE_VALUE) {
      return GetLastError() == ERROR_SHARING_VIOLATION;
    }
    CloseHandle(probe);
    return false;
  };
#else
  struct FileIdentity {
    dev_t dev{};
    ino_t ino{};
  };
  const auto get_identity = [](const std::filesystem::path& path) {
    struct stat st{};
    ORT_ENFORCE(stat(path.c_str(), &st) == 0, "Cannot stat test weight file");
    return FileIdentity{st.st_dev, st.st_ino};
  };
  const auto id_a = get_identity(path_a);
  const auto id_provisional = get_identity(path_provisional);
  const auto id_b = get_identity(path_b);
  const auto count_open_fds = [](const FileIdentity& target) {
    int count = 0;
    for (int fd = 0; fd < 1024; ++fd) {
      struct stat st{};
      if (fstat(fd, &st) == 0 && st.st_dev == target.dev && st.st_ino == target.ino) {
        ++count;
      }
    }
    return count;
  };
#endif

  const char* weight_name = "weights.bin";
  Ort::SessionOptions options;
  {
#ifdef _WIN32
    auto file_a = open_unshared_delete_file(path_a);
    const OrtFileHandle handle_a = file_a.get();
#else
    auto file_a = OpenReadOnlyInitializer(path_a);
    const OrtFileHandle handle_a = NativeFileHandle(file_a.get());
#endif
    Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &weight_name, &handle_a, 1));
  }
#ifdef _WIN32
  EXPECT_TRUE(has_open_handles(path_a));
#else
  EXPECT_EQ(count_open_fds(id_a), 1);
#endif

  auto clone_a = options.Clone();
#ifdef _WIN32
  EXPECT_TRUE(has_open_handles(path_a));
#else
  EXPECT_EQ(count_open_fds(id_a), 1);
#endif

  // Attempt a 2-entry registration where entry 0 duplicates a valid handle and entry 1 fails.
  {
#ifdef _WIN32
    auto file_provisional = open_unshared_delete_file(path_provisional);
    const OrtFileHandle handle_provisional = file_provisional.get();
#else
    auto file_provisional = OpenReadOnlyInitializer(path_provisional);
    const OrtFileHandle handle_provisional = NativeFileHandle(file_provisional.get());
#endif
    const char* batch_names[]{weight_name, "second.bin"};
    const OrtFileHandle batch_handles[]{handle_provisional, kInvalidFileHandle};
    Ort::Status batch_status{
        api_->SessionOptionsSetExternalInitializerFileHandles(options, batch_names, batch_handles, 2)};
    EXPECT_FALSE(batch_status.IsOK());
  }
  // Provisional duplicate from entry 0 must be deterministically released upon batch failure.
#ifdef _WIN32
  EXPECT_FALSE(has_open_handles(path_provisional));
  EXPECT_TRUE(has_open_handles(path_a));
#else
  EXPECT_EQ(count_open_fds(id_provisional), 0);
  EXPECT_EQ(count_open_fds(id_a), 1);
#endif

  const auto serialized = MakeExternalInitializerModel(1, 0).SerializeAsString();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rollback-and-replace-file-sources"};
  const auto run_single_weight = [&](const Ort::SessionOptions& session_options) {
    Ort::Session session{env, serialized.data(), serialized.size(), session_options};
    const char* outputs[]{"Y"};
    auto result = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 1);
    return result.at(0).GetTensorData<float>()[0];
  };
  EXPECT_FLOAT_EQ(run_single_weight(options), 10.0f);
  EXPECT_FLOAT_EQ(run_single_weight(clone_a), 10.0f);

  // Replace options with distinct nonempty registry B while clone_a still retains registry A.
  {
#ifdef _WIN32
    auto file_b = open_unshared_delete_file(path_b);
    const OrtFileHandle handle_b = file_b.get();
#else
    auto file_b = OpenReadOnlyInitializer(path_b);
    const OrtFileHandle handle_b = NativeFileHandle(file_b.get());
#endif
    Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &weight_name, &handle_b, 1));
  }
#ifdef _WIN32
  EXPECT_TRUE(has_open_handles(path_a));
  EXPECT_TRUE(has_open_handles(path_b));
#else
  EXPECT_EQ(count_open_fds(id_a), 1);
  EXPECT_EQ(count_open_fds(id_b), 1);
#endif

  EXPECT_FLOAT_EQ(run_single_weight(options), 30.0f);
  EXPECT_FLOAT_EQ(run_single_weight(clone_a), 10.0f);

  // Releasing clone_a must deterministically release registry A's handle while B remains active.
  clone_a = Ort::SessionOptions{nullptr};
#ifdef _WIN32
  EXPECT_FALSE(has_open_handles(path_a));
  EXPECT_TRUE(has_open_handles(path_b));
#else
  EXPECT_EQ(count_open_fds(id_a), 0);
  EXPECT_EQ(count_open_fds(id_b), 1);
#endif

  // Releasing options must deterministically release registry B's handle.
  options = Ort::SessionOptions{nullptr};
#ifdef _WIN32
  EXPECT_FALSE(has_open_handles(path_b));
#else
  EXPECT_EQ(count_open_fds(id_b), 0);
#endif
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRejectExternalDataInGraphListAttributes) {
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  const float value = 1.0f;
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    ASSERT_TRUE(output.good());
  }

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "graph-list-file-sources"};
  for (bool use_registered_handle : {false, true}) {
    for (bool nested_graph_descendant : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "registered=" << use_registered_handle
                   << ", nested=" << nested_graph_descendant);
      auto model = MakeFileSourceTestModel();
      auto* attr = model.mutable_graph()->mutable_node(0)->add_attribute();
      attr->set_name("unexpected_graphs");
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS);
      const std::string ext_location = use_registered_handle ? "registered_only.bin" : "testdata/Pads.bin";
      if (!nested_graph_descendant) {
        auto subgraph = MakeExternalInitializerModel(1, 0);
        subgraph.mutable_graph()->mutable_initializer(0)->mutable_external_data(0)->set_value(ext_location);
        attr->add_graphs()->CopyFrom(subgraph.graph());
      } else {
        auto nested = MakeNestedFileSourceModel(1, 1, ext_location);
        attr->add_graphs()->CopyFrom(nested.graph());
      }

      Ort::SessionOptions options;
      if (use_registered_handle) {
        auto file = OpenReadOnlyInitializer(path);
        const auto handle = NativeFileHandle(file.get());
        const char* registered_name = "registered_only.bin";
        Ort::ThrowOnError(
            api_->SessionOptionsSetExternalInitializerFileHandles(options, &registered_name, &handle, 1));
      } else {
        Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 0));
      }

      const auto serialized = model.SerializeAsString();
      const auto status = TryCreateSessionFromArray(api_, env, serialized.data(), serialized.size(), options);
      ASSERT_FALSE(status.IsOK())
          << "External initializers inside GRAPHS attributes must be rejected during prevalidation";
      EXPECT_NE(status.GetErrorMessage().find(
                    "File-source loading does not support external data in graph-list attributes."),
                std::string::npos)
          << status.GetErrorMessage();
    }
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRejectExternalSubgraphInitializersWithEmbeddedData) {
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  const std::vector<float> values(64, 2.5f);
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    ASSERT_TRUE(output.good());
  }

  Ort::SessionOptions options;
  {
    auto file = OpenReadOnlyInitializer(path);
    const auto handle = NativeFileHandle(file.get());
    const char* registered_name = "weights.bin";
    Ort::ThrowOnError(api_->SessionOptionsSetExternalInitializerFileHandles(options, &registered_name, &handle, 1));
  }

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "embedded-conflict-file-sources"};
  // Test both <= 127 bytes (4 floats = 16 bytes, inline threshold) and > 127 bytes (64 floats = 256 bytes, mapped).
  for (int64_t length : {int64_t{4}, int64_t{64}}) {
    for (bool use_loop : {false, true}) {
      const auto make_base_model = [&]() {
        return use_loop ? MakeFileSourceLoopModel(length, "weights.bin")
                        : MakeNestedFileSourceModel(length, 1, "weights.bin");
      };
      const auto get_subgraph_initializer = [&](ONNX_NAMESPACE::ModelProto& model) -> ONNX_NAMESPACE::TensorProto* {
        auto* node = model.mutable_graph()->mutable_node(0);
        return node->mutable_attribute(0)->mutable_g()->mutable_initializer(0);
      };

      // 1. Explicitly present but empty raw_data is valid in ONNX and must still succeed.
      {
        auto valid_model = make_base_model();
        get_subgraph_initializer(valid_model)->set_raw_data("");
        const auto serialized = valid_model.SerializeAsString();
        EXPECT_TRUE(TryCreateSessionFromArray(api_, env, serialized.data(), serialized.size(), options).IsOK());
      }

      // 2. Nonempty raw_data on an EXTERNAL initializer must be rejected before replacement.
      {
        auto raw_conflict_model = make_base_model();
        get_subgraph_initializer(raw_conflict_model)
            ->set_raw_data(std::string(static_cast<size_t>(length) * sizeof(float), '\0'));
        const auto serialized = raw_conflict_model.SerializeAsString();
        const auto status = TryCreateSessionFromArray(api_, env, serialized.data(), serialized.size(), options);
        ASSERT_FALSE(status.IsOK()) << "Conflicting raw_data on EXTERNAL subgraph initializer must be rejected";
        EXPECT_NE(status.GetErrorMessage().find("is stored externally and should not have data field.raw_data"),
                  std::string::npos)
            << status.GetErrorMessage();
      }

      // 3. Nonempty typed embedded data fields on an EXTERNAL initializer must be rejected before replacement.
      for (const char* field_name :
           {"float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data"}) {
        SCOPED_TRACE(field_name);
        auto typed_conflict_model = make_base_model();
        auto* tensor = get_subgraph_initializer(typed_conflict_model);
        if (std::string_view{field_name} == "float_data") {
          tensor->add_float_data(99.0f);
        } else if (std::string_view{field_name} == "int32_data") {
          tensor->add_int32_data(99);
        } else if (std::string_view{field_name} == "string_data") {
          tensor->add_string_data("conflict");
        } else if (std::string_view{field_name} == "int64_data") {
          tensor->add_int64_data(99);
        } else if (std::string_view{field_name} == "double_data") {
          tensor->add_double_data(99.0);
        } else {
          tensor->add_uint64_data(99);
        }
        const auto serialized = typed_conflict_model.SerializeAsString();
        const auto status = TryCreateSessionFromArray(api_, env, serialized.data(), serialized.size(), options);
        ASSERT_FALSE(status.IsOK())
            << "Conflicting " << field_name << " on EXTERNAL subgraph initializer must be rejected";
        EXPECT_NE(status.GetErrorMessage().find(
                      std::string{"is stored externally and should not have data field."} + field_name),
                  std::string::npos)
            << status.GetErrorMessage();
      }
    }
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesSupportAddExternalInitializersOverridesForSmallAndLargeWeights) {
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  const std::vector<float> file_values(64, 1.0f);
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(file_values.data()),
                 static_cast<std::streamsize>(file_values.size() * sizeof(float)));
    ASSERT_TRUE(output.good());
  }

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "override-file-sources"};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (int64_t length : {int64_t{4}, int64_t{64}}) {
    SCOPED_TRACE(length);
    Ort::SessionOptions options;
    {
      auto file = OpenReadOnlyInitializer(path);
      const auto handle = NativeFileHandle(file.get());
      const char* registered_name = "weights.bin";
      Ort::ThrowOnError(
          api_->SessionOptionsSetExternalInitializerFileHandles(options, &registered_name, &handle, 1));
    }

    std::vector<float> override_values(static_cast<size_t>(length), 5.0f);
    const int64_t shape[]{length};
    auto override_tensor = Ort::Value::CreateTensor<float>(
        memory, override_values.data(), override_values.size(), shape, 1);
    std::vector<std::string> names{"X"};
    std::vector<Ort::Value> tensors;
    tensors.push_back(std::move(override_tensor));
    options.AddExternalInitializers(names, tensors);

    const auto serialized = MakeExternalInitializerModel(length, 0).SerializeAsString();
    Ort::Session session{env, serialized.data(), serialized.size(), options};
    const char* outputs[]{"Y"};
    auto actual = session.Run(Ort::RunOptions{}, nullptr, nullptr, 0, outputs, 1);
    ASSERT_EQ(actual.size(), 1u);
    const auto* actual_data = actual[0].GetTensorData<float>();
    for (int64_t i = 0; i < length; ++i) {
      EXPECT_FLOAT_EQ(actual_data[i], 5.0f);
    }

    // Overriding a non-external (embedded) initializer must still be rejected.
    auto embedded_model = MakeFileSourceTestModel(length);
    embedded_model.mutable_graph()->clear_input();
    auto* embedded_init = embedded_model.mutable_graph()->add_initializer();
    embedded_init->set_name("X");
    embedded_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    embedded_init->add_dims(length);
    for (int64_t i = 0; i < length; ++i) {
      embedded_init->add_float_data(2.0f);
    }
    const auto embedded_serialized = embedded_model.SerializeAsString();
    const auto bad_status =
        TryCreateSessionFromArray(api_, env, embedded_serialized.data(), embedded_serialized.size(), options);
    ASSERT_FALSE(bad_status.IsOK()) << "AddExternalInitializers must not replace an embedded non-external initializer";
    EXPECT_NE(bad_status.GetErrorMessage().find("Trying to replace non-external initializer with external data"),
              std::string::npos)
        << bad_status.GetErrorMessage();
  }
}

TEST_F(FileSourcesSessionOptionsTest, FileSourcesRejectedInModelPackageOptionsAndSessionCreation) {
  ScopedInitializerDirectory directory;
  const auto weight_path = directory.path / "weights.bin";
  const float weight_value = 3.0f;
  {
    std::ofstream output(weight_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&weight_value), sizeof(weight_value));
    ASSERT_TRUE(output.good());
  }

  const auto package_root = directory.path / "pkg";
  const auto variant_dir = package_root / "main" / "cpu";
  std::filesystem::create_directories(variant_dir);
  {
    std::ofstream model_out(variant_dir / "model.onnx", std::ios::binary);
    ASSERT_TRUE(MakeFileSourceTestModel().SerializeToOstream(&model_out));
    ASSERT_TRUE(model_out.good());
  }
  {
    std::ofstream manifest_out(package_root / "manifest.json", std::ios::binary);
    manifest_out << R"({
  "schema_version": "1.0",
  "components": {
    "main": {
      "variants": {
        "cpu": {
          "variant_directory": "main/cpu",
          "ep": "CPUExecutionProvider",
          "executor_info": {
            "ort": {
              "model_file": "model.onnx"
            }
          }
        }
      }
    }
  }
})";
    ASSERT_TRUE(manifest_out.good());
  }

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "model-package-file-sources"};
  const OrtModelPackageApi* package_api = api_->GetModelPackageApi();
  ASSERT_NE(package_api, nullptr);

  OrtModelPackageOptions* clean_mp_options = nullptr;
  Ort::SessionOptions clean_options;
  Ort::ThrowOnError(package_api->CreateModelPackageOptionsFromSessionOptions(env, clean_options, &clean_mp_options));
  std::unique_ptr<OrtModelPackageOptions, decltype(package_api->ReleaseModelPackageOptions)> mp_options_owner(
      clean_mp_options, package_api->ReleaseModelPackageOptions);

  OrtModelPackageContext* pkg_ctx = nullptr;
  Ort::ThrowOnError(package_api->CreateModelPackageContext(package_root.c_str(), &pkg_ctx));
  std::unique_ptr<OrtModelPackageContext, decltype(package_api->ReleaseModelPackageContext)> pkg_ctx_owner(
      pkg_ctx, package_api->ReleaseModelPackageContext);

  OrtModelPackageComponentContext* comp_ctx = nullptr;
  Ort::ThrowOnError(package_api->SelectComponent(pkg_ctx, "main", clean_mp_options, &comp_ctx));
  std::unique_ptr<OrtModelPackageComponentContext, decltype(package_api->ReleaseModelPackageComponentContext)>
      comp_ctx_owner(comp_ctx, package_api->ReleaseModelPackageComponentContext);

  for (bool empty_registry : {true, false}) {
    SCOPED_TRACE(empty_registry);
    Ort::SessionOptions file_source_options;
    if (empty_registry) {
      Ort::ThrowOnError(
          api_->SessionOptionsSetExternalInitializerFileHandles(file_source_options, nullptr, nullptr, 0));
    } else {
      auto file = OpenReadOnlyInitializer(weight_path);
      const auto handle = NativeFileHandle(file.get());
      const char* registered_name = "weights.bin";
      Ort::ThrowOnError(
          api_->SessionOptionsSetExternalInitializerFileHandles(file_source_options, &registered_name, &handle, 1));
    }

    OrtModelPackageOptions* rejected_mp_options = nullptr;
    Ort::Status options_status{
        package_api->CreateModelPackageOptionsFromSessionOptions(env, file_source_options, &rejected_mp_options)};
    EXPECT_FALSE(options_status.IsOK());
    EXPECT_EQ(options_status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    EXPECT_NE(std::string{options_status.GetErrorMessage()}.find(
                  "External initializer file handles are not supported with model packages."),
              std::string::npos)
        << options_status.GetErrorMessage();
    EXPECT_EQ(rejected_mp_options, nullptr);

    OrtSession* rejected_session = nullptr;
    Ort::Status session_status{
        package_api->CreateSession(env, comp_ctx, file_source_options, &rejected_session)};
    EXPECT_FALSE(session_status.IsOK());
    EXPECT_EQ(session_status.GetErrorCode(), ORT_INVALID_ARGUMENT);
    EXPECT_NE(std::string{session_status.GetErrorMessage()}.find(
                  "External initializer file handles are not supported with model packages."),
              std::string::npos)
        << session_status.GetErrorMessage();
    EXPECT_EQ(rejected_session, nullptr);
  }
}
#endif

#if defined(ORT_MINIMAL_BUILD) || (!defined(_WIN32) && !defined(__linux__) && !defined(__ANDROID__) && !defined(__APPLE__)) || defined(DISABLE_EXTERNAL_INITIALIZERS)
TEST_F(FileSourcesSessionOptionsTest, FileSourcesRejectUnsupportedBuilds) {
  Ort::SessionOptions options;
  Ort::Status status{api_->SessionOptionsSetExternalInitializerFileHandles(options, nullptr, nullptr, 0)};
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_NOT_IMPLEMENTED);
}
#endif
