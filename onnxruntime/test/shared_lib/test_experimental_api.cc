// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_experimental_cxx_api.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <chrono>
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
#include <unistd.h>
#endif

#if (defined(_WIN32) || defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)) && !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
#include "core/graph/onnx_protobuf.h"

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
    if (!std::filesystem::create_directory(path)) {
      throw std::runtime_error("Cannot create initializer test directory");
    }
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
  if (error != 0) {
    throw std::runtime_error("Cannot open initializer test file");
  }
#else
  file = fopen(path.c_str(), "rb");
#endif
  if (file == nullptr) {
    throw std::runtime_error("Cannot open initializer test file");
  }
  return {file, fclose};
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

class ExperimentalCApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ASSERT_NE(api_, nullptr);
  }

  const OrtApi* api_ = nullptr;
};

// Null name returns nullptr
TEST_F(ExperimentalCApiTest, NullNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction(nullptr);
  EXPECT_EQ(fn, nullptr);
}

// Unknown name returns nullptr
TEST_F(ExperimentalCApiTest, UnknownNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction("NonExistentFunction");
  EXPECT_EQ(fn, nullptr);
}

// Empty string returns nullptr
TEST_F(ExperimentalCApiTest, EmptyNameReturnsNull) {
  OrtExperimentalFnPtr fn = api_->GetExperimentalFunction("");
  EXPECT_EQ(fn, nullptr);
}

// Known name resolves to non-null (C-style lookup)
TEST_F(ExperimentalCApiTest, KnownNameResolvesC) {
  OrtExperimentalFnPtr fn =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  EXPECT_NE(fn, nullptr);
}

// Known name resolves to non-null (C++-style accessor).
// The typed accessor's nullptr path (unknown name) is covered transitively via the C-side
// UnknownNameReturnsNull test, which exercises the same underlying GetExperimentalFunction lookup.
TEST_F(ExperimentalCApiTest, KnownNameResolvesCpp) {
  auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api_);
  EXPECT_NE(fn, nullptr);
}

// Call through typed pointer succeeds and returns the expected sentinel value.
// Uses the throwing accessor (FnOrThrow), which returns a guaranteed-non-null pointer or throws.
// The throw-on-unavailable path is not directly unit-testable: every .inc entry is registered, so a valid
// typed accessor never observes a nullptr. Availability checking is covered by the nullable-accessor tests above.
TEST_F(ExperimentalCApiTest, CallThroughTypedPointer) {
  auto* fn = Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_FnOrThrow(api_);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(fn, Ort::Experimental::Get_OrtApi_ExperimentalApiTest_SinceV28_Fn(api_));

  int64_t result = 0;
  auto status = Ort::Status{fn(&result)};
  ASSERT_TRUE(status.IsOK()) << "ExperimentalApiTest returned failure status: " << status.GetErrorMessage();
  EXPECT_EQ(result, 12345);
}

// The same name looked up twice returns the same pointer
TEST_F(ExperimentalCApiTest, ConsistentLookup) {
  OrtExperimentalFnPtr fn1 =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  OrtExperimentalFnPtr fn2 =
      api_->GetExperimentalFunction(kOrtExperimental_OrtApi_ExperimentalApiTest_SinceV28_FnName);
  EXPECT_EQ(fn1, fn2);
}

#if (defined(_WIN32) || defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)) && !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
TEST_F(ExperimentalCApiTest, FileSourcesValidateArgumentsAndAccess) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  Ort::SessionOptions options;
  EXPECT_FALSE(Ort::Status(fn(nullptr, nullptr, nullptr, 0)).IsOK());
  EXPECT_FALSE(Ort::Status(fn(options, nullptr, nullptr, 1)).IsOK());
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
  EXPECT_TRUE(Ort::Status(fn(options, &name, &descriptor, 1)).IsOK());
#else
  EXPECT_FALSE(Ort::Status(fn(options, &name, &descriptor, 1)).IsOK());
#endif
  const auto invalid = kInvalidFileHandle;
  EXPECT_FALSE(Ort::Status(fn(options, &name, &invalid, 1)).IsOK());
  for (const char* bad : {"../weights.data", "/weights.data", ".", "a/b"}) {
    EXPECT_FALSE(Ort::Status(fn(options, &bad, &invalid, 1)).IsOK());
  }
#ifdef O_PATH
  const int path_only = open("testdata/Pads.bin", O_PATH | O_CLOEXEC);
  ASSERT_GE(path_only, 0);
  EXPECT_FALSE(Ort::Status(fn(options, &name, &path_only, 1)).IsOK());
  EXPECT_EQ(close(path_only), 0);
#endif
}

TEST_F(ExperimentalCApiTest, FileSourcesCloneAndRejectBufferMixing) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  Ort::SessionOptions options;
  ASSERT_TRUE(Ort::Status(fn(options, nullptr, nullptr, 0)).IsOK());
  auto clone = options.Clone();
  char data[8]{};
  char* buffer = data;
  const ORTCHAR_T* name = ORT_TSTR("weights.data");
  const size_t size = sizeof(data);
  EXPECT_FALSE(Ort::Status(api_->AddExternalInitializersFromFilesInMemory(
                               clone, &name, &buffer, &size, 1))
                   .IsOK());
}

TEST_F(ExperimentalCApiTest, FileSourcesInferAfterHandleAndOptionsRelease) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
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
  ASSERT_TRUE(Ort::Status(fn(options, &name, &descriptor, 1)).IsOK());
  auto clone = options.Clone();
  weights.reset();
  options = Ort::SessionOptions{nullptr};
  const auto invalid = kInvalidFileHandle;
  EXPECT_FALSE(Ort::Status(fn(clone, &name, &invalid, 1)).IsOK());

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

TEST_F(ExperimentalCApiTest, FileSourcesNeverFallBackToModelPaths) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  std::ifstream input("testdata/model_with_external_initializers.onnx", std::ios::binary);
  ASSERT_TRUE(input);
  const std::vector<char> graph{std::istreambuf_iterator<char>{input}, {}};
  ASSERT_FALSE(graph.empty());
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "file-sources"};
  Ort::SessionOptions options;
  ASSERT_TRUE(Ort::Status(fn(options, nullptr, nullptr, 0)).IsOK());
  try {
    Ort::Session session{env, graph.data(), graph.size(), options};
    FAIL() << "Missing registered sources must not fall back to disk";
  } catch (const Ort::Exception& error) {
    EXPECT_NE(std::string(error.what()).find("No registered external initializer file source"),
              std::string::npos);
    EXPECT_NE(std::string(error.what()).find("Pads.bin"), std::string::npos);
  }
}

TEST_F(ExperimentalCApiTest, FileSourcesRejectExternalSparseDataBeforeGraphConstruction) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
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
      Ort::ThrowOnError(fn(options, nullptr, nullptr, 0));
      const auto bytes = model.SerializeAsString();
      try {
        Ort::Session session{env, bytes.data(), bytes.size(), options};
        FAIL() << "External sparse data must be rejected before graph construction";
      } catch (const Ort::Exception& error) {
        EXPECT_NE(std::string{error.what()}.find("does not support external sparse tensors"), std::string::npos);
      }
    }
  }
}

TEST_F(ExperimentalCApiTest, FileSourcesRejectExternalDataInModelLocalFunctions) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
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
    Ort::ThrowOnError(fn(options, nullptr, nullptr, 0));
    const auto bytes = model.SerializeAsString();
    try {
      Ort::Session session{env, bytes.data(), bytes.size(), options};
      FAIL() << "External function data must not reach pathname loading";
    } catch (const Ort::Exception& error) {
      const std::string message{error.what()};
      EXPECT_TRUE(message.find("does not support external tensor attributes") != std::string::npos ||
                  message.find("does not support external data in model-local functions") != std::string::npos ||
                  message.find("does not support external sparse tensors") != std::string::npos)
          << message;
    }
  }
}

TEST_F(ExperimentalCApiTest, FileSourcesValidateZeroLengthRanges) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  ScopedInitializerDirectory directory;
  const auto path = directory.path / "weights.bin";
  {
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.put('x');
  }
  auto file = OpenReadOnlyInitializer(path);
  const auto handle = NativeFileHandle(file.get());
  const char* name = "weights.bin";
  Ort::SessionOptions options;
  Ort::ThrowOnError(fn(options, &name, &handle, 1));
  file.reset();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "empty-file-sources"};
  const auto invalid = MakeExternalInitializerModel(0, 2).SerializeAsString();
  try {
    Ort::Session session{env, invalid.data(), invalid.size(), options};
    FAIL() << "An empty initializer cannot begin beyond EOF";
  } catch (const Ort::Exception& error) {
    EXPECT_NE(std::string{error.what()}.find("Invalid registered external initializer range"), std::string::npos);
  }
  const auto valid = MakeExternalInitializerModel(0, 1).SerializeAsString();
  EXPECT_NO_THROW(Ort::Session(env, valid.data(), valid.size(), options));
}

TEST_F(ExperimentalCApiTest, FileSourcesRejectHiddenDuplicateExternalInitializers) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
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
    Ort::ThrowOnError(fn(options, nullptr, nullptr, 0));
    const auto bytes = model.SerializeAsString();
    try {
      Ort::Session session{env, bytes.data(), bytes.size(), options};
      FAIL() << "Duplicate names must not hide an external initializer from the registry";
    } catch (const Ort::Exception& error) {
      EXPECT_NE(std::string{error.what()}.find("duplicate initializer names: X"), std::string::npos);
    }
  }
}

TEST_F(ExperimentalCApiTest, FileSourcesRetainLargeAlignedAndUnalignedInitializers) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  const std::vector<float> expected(64, 3.25f);
  ASSERT_GT(expected.size() * sizeof(float), 127u) << "Exercise ownership above TensorProto's inline limit";
  for (int64_t offset : {0, 1}) {
    SCOPED_TRACE(offset);
    ScopedInitializerDirectory directory;
    const auto path = directory.path / "weights.bin";
    {
      std::ofstream output(path, std::ios::binary);
      output.exceptions(std::ios::badbit | std::ios::failbit);
      if (offset != 0) output.put('x');
      output.write(reinterpret_cast<const char*>(expected.data()),
                   static_cast<std::streamsize>(expected.size() * sizeof(float)));
    }
    auto file = OpenReadOnlyInitializer(path);
    const auto handle = NativeFileHandle(file.get());
    const char* name = "weights.bin";
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    Ort::ThrowOnError(fn(options, &name, &handle, 1));
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
#endif

#if defined(ORT_MINIMAL_BUILD)
TEST_F(ExperimentalCApiTest, FileSourcesAreNotAdvertisedInMinimalBuilds) {
  EXPECT_EQ(api_->GetExperimentalFunction(
                "OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32"),
            nullptr);
}
#elif (!defined(_WIN32) && !defined(__linux__) && !defined(__ANDROID__) && !defined(__APPLE__)) || defined(DISABLE_EXTERNAL_INITIALIZERS)
TEST_F(ExperimentalCApiTest, FileSourcesRejectUnsupportedBuilds) {
  auto* fn = Ort::Experimental::
      Get_OrtApi_SessionOptionsSetExternalInitializerFileHandles_SinceV32_FnOrThrow(api_);
  Ort::SessionOptions options;
  Ort::Status status{fn(options, nullptr, nullptr, 0)};
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_NOT_IMPLEMENTED);
}
#endif
