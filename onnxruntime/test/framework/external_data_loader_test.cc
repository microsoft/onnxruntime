// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader_manager.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(__wasm__) && !defined(ORT_NO_EXCEPTIONS)
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

#include "core/framework/external_data_loader_thread_pool.h"
#endif

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)

#include <algorithm>
#include <cstdio>
#include <string>

#include "core/common/inlined_containers.h"
#include "core/common/path_string.h"
#include "core/framework/execution_providers.h"
#include "core/framework/external_data_loader.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#ifdef _WIN32
#include <Windows.h>
#endif
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/file_util.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
enum class ReadFailure { None,
                         Status,
                         Exception };

struct LoaderState {
  size_t created{0};
  size_t destroyed{0};
  bool fail_creation{false};
  ReadFailure failure{ReadFailure::None};
  InlinedVector<FileOffsetType> offsets;
  ExternalDataLoaderOptions configured_options{};
  bool is_enabled{true};
  bool defer_reads{false};
  size_t begin_count{0};
  size_t finalize_count{0};
  size_t end_count{0};
  size_t expected_load_bytes{0};
  struct DeferredRead {
    const RandomAccessFile* file{nullptr};
    FileOffsetType offset{0};
    size_t length{0};
    char* destination{nullptr};
  };
  std::vector<DeferredRead> pending_reads;
  std::vector<size_t> batch_sizes;
};

class TrackingExternalDataLoader final : public IExternalDataLoader {
 public:
  explicit TrackingExternalDataLoader(std::shared_ptr<LoaderState> state) : state_(std::move(state)) {
    ++state_->created;
  }
  ~TrackingExternalDataLoader() override { ++state_->destroyed; }

  bool CanLoad(const OrtMemoryInfo& memory_info) const override {
    return memory_info.device.Type() == OrtDevice::CPU;
  }

  bool IsEnabled() const noexcept override {
    return state_->is_enabled;
  }

  common::Status Configure(const ExternalDataLoaderOptions& options) const override {
    state_->configured_options = options;
    if (options.reading_threads.has_value() && *options.reading_threads == 0) {
      state_->is_enabled = false;
    }
    return common::Status::OK();
  }

  common::Status BeginLoad() const override {
    ++state_->begin_count;
    state_->pending_reads.clear();
    return common::Status::OK();
  }

  void SetExpectedLoadBytes(size_t total_bytes) const override {
    state_->expected_load_bytes = total_bytes;
  }

  common::Status FinalizeLoad(const std::function<bool()>&) const override {
    ++state_->finalize_count;
#if !defined(__wasm__)
    if (!state_->pending_reads.empty()) {
      state_->batch_sizes.push_back(state_->pending_reads.size());
      for (const auto& pending : state_->pending_reads) {
        ORT_RETURN_IF_ERROR(
            pending.file->Read(pending.offset, gsl::span<char>(pending.destination, pending.length)));
      }
      state_->pending_reads.clear();
    }
#endif
    return common::Status::OK();
  }

  void EndLoad() const noexcept override {
    ++state_->end_count;
    state_->pending_reads.clear();
  }

#if defined(__wasm__)
  Status LoadTensor(const Env& env, const std::filesystem::path& path, FileOffsetType offset,
                    SafeInt<size_t> length, Tensor& tensor) const override {
#else
  Status LoadTensor(const RandomAccessFile& file, FileOffsetType offset,
                    SafeInt<size_t> length, Tensor& tensor) const override {
#endif
    state_->offsets.push_back(offset);
    if (state_->failure == ReadFailure::Exception) {
      ORT_THROW("external loader read exception");
    }
    ORT_RETURN_IF(state_->failure == ReadFailure::Status, "external loader read failure");
    ORT_RETURN_IF_NOT(length == tensor.SizeInBytes(), "external data length does not match tensor size");
#if defined(__wasm__)
    return env.ReadFileIntoBuffer(
        path.c_str(), offset, length,
        gsl::span<char>(static_cast<char*>(tensor.MutableDataRaw()), tensor.SizeInBytes()));
#else
    if (state_->defer_reads) {
      state_->pending_reads.push_back(LoaderState::DeferredRead{
          &file, offset, tensor.SizeInBytes(), static_cast<char*>(tensor.MutableDataRaw())});
      return common::Status::OK();
    }
    return file.Read(offset,
                     gsl::span<char>(static_cast<char*>(tensor.MutableDataRaw()), tensor.SizeInBytes()));
#endif
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TrackingExternalDataLoader);
  std::shared_ptr<LoaderState> state_;
};

class CPUExecutionProviderWithLoader final : public CPUExecutionProvider {
 public:
  explicit CPUExecutionProviderWithLoader(std::shared_ptr<LoaderState> state)
      : CPUExecutionProvider(CPUExecutionProviderInfo{}), state_(std::move(state)) {}

  std::unique_ptr<IExternalDataLoader> GetExternalDataLoader() const override {
    auto loader = std::make_unique<TrackingExternalDataLoader>(state_);
    if (state_->fail_creation) {
      ORT_THROW("external loader creation exception");
    }
    return loader;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CPUExecutionProviderWithLoader);
  std::shared_ptr<LoaderState> state_;
};

void SetBoolType(ONNX_NAMESPACE::ValueInfoProto& value, const char* name, bool scalar = false) {
  value.set_name(name);
  auto* type = value.mutable_type()->mutable_tensor_type();
  type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  auto* shape = type->mutable_shape();
  if (!scalar) {
    shape->add_dim()->set_dim_value(1);
  }
}

void AddExternalWeight(ONNX_NAMESPACE::GraphProto& graph, const char* name,
                       const PathString& data_path, size_t offset) {
  auto* weight = graph.add_initializer();
  weight->set_name(name);
  weight->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  weight->add_dims(1);
  weight->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* location = weight->add_external_data();
  location->set_key("location");
  location->set_value(ToUTF8String(data_path));
  auto* offset_entry = weight->add_external_data();
  offset_entry->set_key("offset");
  offset_entry->set_value(std::to_string(offset));
  auto* length = weight->add_external_data();
  length->set_key("length");
  length->set_value("1");
}

ONNX_NAMESPACE::ModelProto MakeModel(const PathString& data_path, bool with_subgraphs) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model.add_opset_import()->set_version(13);
  auto& graph = *model.mutable_graph();
  graph.set_name("external_loader_lifetime");
  SetBoolType(*graph.add_input(), "input");
  SetBoolType(*graph.add_output(), "output");
  AddExternalWeight(graph, "weight", data_path, 0);
  auto* node = graph.add_node();
  node->set_op_type("And");
  node->add_input("input");
  node->add_input("weight");
  node->add_output(with_subgraphs ? "outer" : "output");
  if (with_subgraphs) {
    SetBoolType(*graph.add_input(), "condition", true);
    auto* if_node = graph.add_node();
    if_node->set_op_type("If");
    if_node->add_input("condition");
    if_node->add_output("output");
    for (const bool then_branch : {true, false}) {
      auto* attribute = if_node->add_attribute();
      attribute->set_name(then_branch ? "then_branch" : "else_branch");
      attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
      auto& branch = *attribute->mutable_g();
      branch.set_name(attribute->name());
      SetBoolType(*branch.add_output(), "branch_output");
      AddExternalWeight(branch, "branch_weight", data_path, then_branch ? 1 : 2);
      auto* branch_node = branch.add_node();
      branch_node->set_op_type("Or");
      branch_node->add_input("outer");
      branch_node->add_input("branch_weight");
      branch_node->add_output("branch_output");
    }
  }
  return model;
}

void WriteTestFile(const std::string& bytes, PathString& path, ScopedFileDeleter& deleter) {
  FILE* file = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateTestFile(file, path));
  deleter = ScopedFileDeleter(path);
  std::unique_ptr<FILE, int (*)(FILE*)> file_owner(file, fclose);
  ASSERT_EQ(bytes.size(), fwrite(bytes.data(), 1, bytes.size(), file));
  ASSERT_EQ(0, fclose(file_owner.release()));
}

class ExternalDataLoaderLifetimeTest : public testing::Test {
 protected:
  void CreateSession(bool with_subgraphs = false) {
    PathString data_path = ORT_TSTR("external_loader_weights_XXXXXX");
    ASSERT_NO_FATAL_FAILURE(WriteTestFile(std::string("\1\1\0", 3), data_path, data_deleter_));
    PathString model_path = ORT_TSTR("external_loader_model_XXXXXX");
    ASSERT_NO_FATAL_FAILURE(
        WriteTestFile(MakeModel(data_path, with_subgraphs).SerializeAsString(), model_path, model_deleter_));
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    options.intra_op_param.thread_pool_size = 1;
    session_ = std::make_unique<InferenceSession>(options, GetEnvironment());
    ASSERT_STATUS_OK(session_->RegisterExecutionProvider(std::make_unique<CPUExecutionProviderWithLoader>(state_)));
    ASSERT_STATUS_OK(session_->Load(model_path));
  }

  void ExpectReleased(size_t count) {
    EXPECT_EQ(state_->created, count);
    EXPECT_EQ(state_->destroyed, count);
    EXPECT_EQ(session_->GetExternalDataLoaderManager().GetExternalDataLoader(OrtMemoryInfo(CPU, OrtDeviceAllocator)),
              nullptr);
  }

  void Run(bool input, bool expected, bool with_subgraphs = false, bool condition = false) {
    OrtValue input_value;
    CreateMLValue<bool>(std::make_shared<CPUAllocator>(), {1}, {input}, &input_value);
    NameMLValMap feeds{{"input", input_value}};
    if (with_subgraphs) {
      OrtValue condition_value;
      CreateMLValue<bool>(std::make_shared<CPUAllocator>(), {}, {condition}, &condition_value);
      feeds.emplace("condition", std::move(condition_value));
    }
    const InlinedVector<std::string> output_names{"output"};
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session_->Run(feeds, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1U);
    ASSERT_EQ(fetches[0].Get<Tensor>().Shape(), TensorShape({1}));
    EXPECT_EQ(fetches[0].Get<Tensor>().Data<bool>()[0], expected);
  }

  void TestFailedInitialization(ReadFailure failure) {
    ASSERT_NO_FATAL_FAILURE(CreateSession());
    state_->failure = failure;
    const auto status = session_->Initialize();
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find("external loader read"), std::string::npos);
    ExpectReleased(1);
    EXPECT_EQ(state_->offsets.size(), 1U);
    session_.reset();
    EXPECT_EQ(state_->destroyed, 1U);
  }

  ScopedFileDeleter data_deleter_;
  ScopedFileDeleter model_deleter_;
  std::shared_ptr<LoaderState> state_{std::make_shared<LoaderState>()};
  std::unique_ptr<InferenceSession> session_;
};

TEST_F(ExternalDataLoaderLifetimeTest, CreatesLoadersOnlyWhenInitializing) {
  ASSERT_NO_FATAL_FAILURE(CreateSession());
  ExpectReleased(0);
  session_.reset();
  EXPECT_EQ(state_->created, 0U);
  EXPECT_EQ(state_->destroyed, 0U);
}

TEST_F(ExternalDataLoaderLifetimeTest, CancellationBeforeInitializationDoesNotCreateLoaders) {
  ASSERT_NO_FATAL_FAILURE(CreateSession());
  session_->GetMutableSessionOptions().SetLoadCancellationFlag(true);
  const auto status = session_->Initialize();
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::MODEL_LOAD_CANCELED);
  ExpectReleased(0);

  session_->GetMutableSessionOptions().SetLoadCancellationFlag(false);
  ASSERT_STATUS_OK(session_->Initialize());
  ExpectReleased(1);
}

TEST_F(ExternalDataLoaderLifetimeTest, ReleasesBeforeSessionDestructionAndDoesNotReloadForInference) {
  ASSERT_NO_FATAL_FAILURE(CreateSession());
  ASSERT_STATUS_OK(session_->Initialize());
  ExpectReleased(1);
  EXPECT_EQ(&session_->GetSessionState().GetExternalDataLoaderMgr(), &session_->GetExternalDataLoaderManager());
  ASSERT_NO_FATAL_FAILURE(Run(false, false));
  ASSERT_NO_FATAL_FAILURE(Run(true, true));
  ASSERT_STATUS_OK(session_->Initialize());
  ExpectReleased(1);
  EXPECT_EQ(state_->offsets.size(), 1U);
  session_.reset();
  EXPECT_EQ(state_->destroyed, 1U);
}

TEST_F(ExternalDataLoaderLifetimeTest, KeepsLoaderUntilBothSubgraphsHaveLoaded) {
  ASSERT_NO_FATAL_FAILURE(CreateSession(true));
  ASSERT_STATUS_OK(session_->Initialize());
  ExpectReleased(1);
  std::sort(state_->offsets.begin(), state_->offsets.end());
  EXPECT_EQ(state_->offsets, (InlinedVector<FileOffsetType>{0, 1, 2}));
  ASSERT_NO_FATAL_FAILURE(Run(false, true, true, true));
  ASSERT_NO_FATAL_FAILURE(Run(false, false, true, false));
  EXPECT_EQ(state_->offsets.size(), 3U);
}

TEST_F(ExternalDataLoaderLifetimeTest, ReleasesOnReadFailure) {
  TestFailedInitialization(ReadFailure::Status);
}

TEST_F(ExternalDataLoaderLifetimeTest, ConfiguresLoaderFromSessionOptions) {
  ASSERT_NO_FATAL_FAILURE(CreateSession());
  auto& config_options = session_->GetMutableSessionOptions().config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(kOrtSessionOptionsExternalDataLoaderThreads, "8"));
  ASSERT_STATUS_OK(config_options.AddConfigEntry(
      kOrtSessionOptionsExternalDataLoaderPinnedMemoryBytes, "33554432"));
  ASSERT_STATUS_OK(config_options.AddConfigEntry(
      kOrtSessionOptionsExternalDataLoaderChunkBytes, "1048576"));
  ASSERT_STATUS_OK(config_options.AddConfigEntry(
      kOrtSessionOptionsExternalDataLoaderMinSizeBytes, "8388608"));
  ASSERT_STATUS_OK(session_->Initialize());
  ExpectReleased(1);
  EXPECT_EQ(state_->offsets.size(), 1U);
  EXPECT_EQ(state_->configured_options.reading_threads, std::optional<size_t>{8});
  EXPECT_EQ(state_->configured_options.auto_reading_threads,
            std::optional<size_t>{ExternalDataLoaderOptions::GetDefaultReadingThreads()});
  EXPECT_EQ(state_->configured_options.pinned_memory_bytes, std::optional<size_t>{33554432});
  EXPECT_EQ(state_->configured_options.chunk_bytes, std::optional<size_t>{1048576});
  EXPECT_EQ(state_->configured_options.min_size_bytes, std::optional<size_t>{8388608});
  ASSERT_NO_FATAL_FAILURE(Run(true, true));
}

#if !defined(__wasm__)
TEST_F(ExternalDataLoaderLifetimeTest, FlushesDeferredLoadsBeforeClearingExternalFileCache) {
  constexpr size_t kExternalFileCount = 66;  // Exceeds the 64-file cache threshold in session_state_utils.cc.
  std::vector<PathString> data_paths(kExternalFileCount);
  std::vector<ScopedFileDeleter> data_deleters(kExternalFileCount);

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model.add_opset_import()->set_version(13);
  auto& graph = *model.mutable_graph();
  graph.set_name("external_loader_many_files");
  SetBoolType(*graph.add_input(), "input");
  SetBoolType(*graph.add_output(), "output");

  std::string current_input = "input";
  for (size_t i = 0; i < kExternalFileCount; ++i) {
    data_paths[i] = ORT_TSTR("external_loader_many_weights_") + ToPathString(std::to_string(i)) + ORT_TSTR("_XXXXXX");
    std::string file_bytes(i + 1, '\0');
    file_bytes[i] = '\1';
    ASSERT_NO_FATAL_FAILURE(WriteTestFile(file_bytes, data_paths[i], data_deleters[i]));
    const std::string weight_name = "weight_" + std::to_string(i);
    AddExternalWeight(graph, weight_name.c_str(), data_paths[i], i);

    auto* node = graph.add_node();
    node->set_op_type("And");
    node->add_input(current_input);
    node->add_input(weight_name);
    const std::string next_output = (i + 1 == kExternalFileCount) ? "output" : ("inter_" + std::to_string(i));
    node->add_output(next_output);
    current_input = next_output;
  }

  PathString model_path = ORT_TSTR("external_loader_many_model_XXXXXX");
  ASSERT_NO_FATAL_FAILURE(WriteTestFile(model.SerializeAsString(), model_path, model_deleter_));

  state_->defer_reads = true;
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  session_ = std::make_unique<InferenceSession>(options, GetEnvironment());
  ASSERT_STATUS_OK(session_->RegisterExecutionProvider(std::make_unique<CPUExecutionProviderWithLoader>(state_)));
  ASSERT_STATUS_OK(session_->Load(model_path));
  ASSERT_STATUS_OK(session_->Initialize());

  ExpectReleased(1);
  EXPECT_EQ(state_->begin_count, 1U);
  EXPECT_EQ(state_->finalize_count, 2U);
  EXPECT_EQ(state_->end_count, 1U);
  EXPECT_EQ(state_->expected_load_bytes, kExternalFileCount);
  EXPECT_EQ(state_->batch_sizes, (std::vector<size_t>{64, 2}));
  ASSERT_EQ(state_->offsets.size(), kExternalFileCount);
  std::sort(state_->offsets.begin(), state_->offsets.end());
  for (size_t i = 0; i < kExternalFileCount; ++i) {
    EXPECT_EQ(state_->offsets[i], static_cast<FileOffsetType>(i));
  }
  ASSERT_NO_FATAL_FAILURE(Run(true, true));
  ASSERT_NO_FATAL_FAILURE(Run(false, false));
}
#endif

// The default CPU fallback path reads external data through Module.MountedFiles, which is unavailable in WASM tests.
#if !defined(__wasm__)
TEST_F(ExternalDataLoaderLifetimeTest, DisablesLoaderWhenSessionOptionsThreadsIsZero) {
  ASSERT_NO_FATAL_FAILURE(CreateSession());
  ASSERT_STATUS_OK(session_->GetMutableSessionOptions().config_options.AddConfigEntry(
      kOrtSessionOptionsExternalDataLoaderThreads, "0"));
  ASSERT_STATUS_OK(session_->Initialize());
  ExpectReleased(1);
  // Loader was disabled by threads=0, so the framework's default CPU path loaded the tensor.
  EXPECT_TRUE(state_->offsets.empty());
  ASSERT_NO_FATAL_FAILURE(Run(true, true));
}
#endif

TEST_F(ExternalDataLoaderLifetimeTest, RejectsInvalidSessionOptionValues) {
  struct InvalidCase {
    const char* key;
    const char* value;
  };
  const std::array<InvalidCase, 7> invalid_cases{{
      {kOrtSessionOptionsExternalDataLoaderThreads, "65"},
      {kOrtSessionOptionsExternalDataLoaderThreads, "-1"},
      {kOrtSessionOptionsExternalDataLoaderPinnedMemoryBytes, "4096"},
      {kOrtSessionOptionsExternalDataLoaderPinnedMemoryBytes, "1048577"},
      {kOrtSessionOptionsExternalDataLoaderPinnedMemoryBytes, "2147483648"},
      {kOrtSessionOptionsExternalDataLoaderChunkBytes, "2048"},
      {kOrtSessionOptionsExternalDataLoaderChunkBytes, "134217728"},
  }};
  for (const auto& test_case : invalid_cases) {
    SCOPED_TRACE(std::string(test_case.key) + "=" + test_case.value);
    state_ = std::make_shared<LoaderState>();
    ASSERT_NO_FATAL_FAILURE(CreateSession());
    ASSERT_STATUS_OK(session_->GetMutableSessionOptions().config_options.AddConfigEntry(
        test_case.key, test_case.value));
    const auto status = session_->Initialize();
    ASSERT_FALSE(status.IsOK());
    EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
  }
}

#ifndef ORT_NO_EXCEPTIONS
TEST_F(ExternalDataLoaderLifetimeTest, ReleasesOnReadException) {
  TestFailedInitialization(ReadFailure::Exception);
}

TEST_F(ExternalDataLoaderLifetimeTest, RecreatesLoaderAfterFactoryFailure) {
  ASSERT_NO_FATAL_FAILURE(CreateSession());
  state_->fail_creation = true;
  const auto status = session_->Initialize();
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("external loader creation exception"), std::string::npos);
  ExpectReleased(1);
  EXPECT_TRUE(state_->offsets.empty());

  state_->fail_creation = false;
  ASSERT_STATUS_OK(session_->Initialize());
  ExpectReleased(2);
  ASSERT_NO_FATAL_FAILURE(Run(true, true));
}
#endif

#endif

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
TEST(ExternalDataLoaderManagerTest, FileSourceInitializersTrackEligibleRangesUntilReplacedOrCleaned) {
  PathString data_path = ORT_TSTR("external_file_source_ranges_XXXXXX");
  ScopedFileDeleter data_deleter;
  std::string payload(512, '\0');
  ASSERT_NO_FATAL_FAILURE(WriteTestFile(payload, data_path, data_deleter));

  std::unique_ptr<RandomAccessFile> opened_file;
  ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(data_path.c_str(), opened_file));
  std::shared_ptr<RandomAccessFile> shared_file{std::move(opened_file)};
  auto files = std::make_shared<
      std::unordered_map<PathString, std::shared_ptr<RandomAccessFile>>>();
  files->emplace(ORT_TSTR("weights.bin"), shared_file);

  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model_proto.add_opset_import()->set_version(13);
  auto& graph_proto = *model_proto.mutable_graph();
  graph_proto.set_name("file_source_tracking");

  auto add_external_tensor = [&](const char* name, int32_t data_type, int64_t dims,
                                 int64_t offset, int64_t length) {
    auto* tensor = graph_proto.add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(data_type);
    tensor->add_dims(dims);
    tensor->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* loc = tensor->add_external_data();
    loc->set_key("location");
    loc->set_value("weights.bin");
    auto* off = tensor->add_external_data();
    off->set_key("offset");
    off->set_value(std::to_string(offset));
    auto* len = tensor->add_external_data();
    len->set_key("length");
    len->set_value(std::to_string(length));
  };

  add_external_tensor("large_aligned", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 64, 0, 256);
  add_external_tensor("small_aligned", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 8, 256, 32);
  add_external_tensor("unaligned_float", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 64, 1, 256);
  add_external_tensor("bool_tensor", ONNX_NAMESPACE::TensorProto_DataType_BOOL, 16, 300, 16);

  Model model(model_proto, PathString{}, nullptr, DefaultLoggingManager().DefaultLogger(),
              ModelOptions(true, false, {}, files));
  Graph& graph = model.MainGraph();

  Graph::ExternalInitializerFileSource source;
  ASSERT_TRUE(graph.HasExternalInitializerFileSource("large_aligned"));
  ASSERT_TRUE(graph.GetExternalInitializerFileSource("large_aligned", source));
  EXPECT_EQ(source.location, ORT_TSTR("weights.bin"));
  EXPECT_EQ(source.file.get(), shared_file.get());
  EXPECT_EQ(source.offset, 0);
  EXPECT_EQ(source.length, 256u);

  ASSERT_TRUE(graph.HasExternalInitializerFileSource("small_aligned"));
  ASSERT_TRUE(graph.GetExternalInitializerFileSource("small_aligned", source));
  EXPECT_EQ(source.offset, 256);
  EXPECT_EQ(source.length, 32u);

  EXPECT_FALSE(graph.HasExternalInitializerFileSource("unaligned_float"));
  EXPECT_FALSE(graph.HasExternalInitializerFileSource("bool_tensor"));

  graph.RemoveInitializedTensor("small_aligned");
  EXPECT_FALSE(graph.HasExternalInitializerFileSource("small_aligned"));

  graph.CleanAllInitializedTensors();
  EXPECT_FALSE(graph.HasExternalInitializerFileSource("large_aligned"));
}

TEST(ExternalDataLoaderManagerTest, InlineIfSubgraphPreservesFileSourceMetadata) {
  PathString data_path = ORT_TSTR("external_file_source_inline_if_XXXXXX");
  ScopedFileDeleter data_deleter;
  std::string payload(512, '\0');
  ASSERT_NO_FATAL_FAILURE(WriteTestFile(payload, data_path, data_deleter));

  std::unique_ptr<RandomAccessFile> opened_file;
  ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(data_path.c_str(), opened_file));
  std::shared_ptr<RandomAccessFile> shared_file{std::move(opened_file)};
  auto files = std::make_shared<
      std::unordered_map<PathString, std::shared_ptr<RandomAccessFile>>>();
  files->emplace(ORT_TSTR("weights.bin"), shared_file);

  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model_proto.add_opset_import()->set_version(13);
  auto& graph_proto = *model_proto.mutable_graph();
  graph_proto.set_name("inline_if_file_source");

  auto* cond_init = graph_proto.add_initializer();
  cond_init->set_name("cond");
  cond_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  cond_init->add_int32_data(1);

  auto set_float_type = [](ONNX_NAMESPACE::ValueInfoProto* value_info, const char* name, int64_t dim) {
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
  };
  set_float_type(graph_proto.add_input(), "X", 64);
  set_float_type(graph_proto.add_output(), "Y0", 64);
  set_float_type(graph_proto.add_output(), "Y1", 64);

  auto* if_node = graph_proto.add_node();
  if_node->set_op_type("If");
  if_node->set_name("const_if");
  if_node->add_input("cond");
  if_node->add_output("if_direct_out");
  if_node->add_output("if_added_out");

  auto* id0 = graph_proto.add_node();
  id0->set_op_type("Identity");
  id0->add_input("if_direct_out");
  id0->add_output("Y0");
  auto* id1 = graph_proto.add_node();
  id1->set_op_type("Identity");
  id1->add_input("if_added_out");
  id1->add_output("Y1");

  auto add_external_float_tensor = [](ONNX_NAMESPACE::GraphProto& target_graph, const char* name,
                                      int64_t offset, int64_t length) {
    auto* tensor = target_graph.add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor->add_dims(64);
    tensor->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* loc = tensor->add_external_data();
    loc->set_key("location");
    loc->set_value("weights.bin");
    auto* off = tensor->add_external_data();
    off->set_key("offset");
    off->set_value(std::to_string(offset));
    auto* len = tensor->add_external_data();
    len->set_key("length");
    len->set_value(std::to_string(length));
  };

  auto* then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto& then_graph = *then_attr->mutable_g();
  then_graph.set_name("then_graph");
  // direct_out_weight is directly a subgraph output, so InlineIfSubgraph renames it to "if_direct_out".
  add_external_float_tensor(then_graph, "direct_out_weight", 0, 256);
  // internal_branch_weight is consumed by a node inside then_graph, so InlineIfSubgraph renames it via GenerateNodeArgName.
  add_external_float_tensor(then_graph, "internal_branch_weight", 256, 256);
  auto* then_add = then_graph.add_node();
  then_add->set_op_type("Add");
  then_add->add_input("X");
  then_add->add_input("internal_branch_weight");
  then_add->add_output("then_added_out");
  set_float_type(then_graph.add_output(), "direct_out_weight", 64);
  set_float_type(then_graph.add_output(), "then_added_out", 64);

  auto* else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto& else_graph = *else_attr->mutable_g();
  else_graph.set_name("else_graph");
  add_external_float_tensor(else_graph, "else_w0", 0, 256);
  add_external_float_tensor(else_graph, "else_w1", 256, 256);
  set_float_type(else_graph.add_output(), "else_w0", 64);
  set_float_type(else_graph.add_output(), "else_w1", 64);

  Model model(model_proto, PathString{}, nullptr, DefaultLoggingManager().DefaultLogger(),
              ModelOptions(true, false, {}, files));
  Graph& graph = model.MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  Node* if_node_ptr = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "If") {
      if_node_ptr = &node;
      break;
    }
  }
  ASSERT_NE(if_node_ptr, nullptr);
  ASSERT_STATUS_OK(graph.InlineIfSubgraph(/*condition_value=*/true, *if_node_ptr,
                                          DefaultLoggingManager().DefaultLogger()));
  graph_utils::RemoveNodeOutputEdges(graph, *if_node_ptr);
  graph.RemoveNode(if_node_ptr->Index());
  ASSERT_STATUS_OK(graph.Resolve());

  Graph::ExternalInitializerFileSource direct_source;
  ASSERT_TRUE(graph.GetExternalInitializerFileSource("if_direct_out", direct_source));
  EXPECT_EQ(direct_source.location, ORT_TSTR("weights.bin"));
  EXPECT_EQ(direct_source.file.get(), shared_file.get());
  EXPECT_EQ(direct_source.offset, 0);
  EXPECT_EQ(direct_source.length, 256u);

  std::string renamed_internal_weight;
  for (const auto& [name, proto] : graph.GetAllInitializedTensors()) {
    if (name != "cond" && name != "if_direct_out") {
      renamed_internal_weight = name;
      break;
    }
  }
  ASSERT_FALSE(renamed_internal_weight.empty());
  Graph::ExternalInitializerFileSource internal_source;
  ASSERT_TRUE(graph.GetExternalInitializerFileSource(renamed_internal_weight, internal_source));
  EXPECT_EQ(internal_source.location, ORT_TSTR("weights.bin"));
  EXPECT_EQ(internal_source.file.get(), shared_file.get());
  EXPECT_EQ(internal_source.offset, 256);
  EXPECT_EQ(internal_source.length, 256u);

  // Genuine value replacement must still invalidate the file-source record.
  ONNX_NAMESPACE::TensorProto modified_proto = *graph.GetAllInitializedTensors().at(renamed_internal_weight);
  modified_proto.clear_external_data();
  modified_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_DEFAULT);
  modified_proto.set_raw_data(std::string(256, '\0'));
  ASSERT_STATUS_OK(graph.ReplaceInitializedTensor(modified_proto, OrtValue{}));
  EXPECT_FALSE(graph.HasExternalInitializerFileSource(renamed_internal_weight));
}

class DummyGpuExecutionProvider final : public IExecutionProvider {
 public:
  static constexpr const char* kType = "DummyGpuMemcpyTestExecutionProvider";
  DummyGpuExecutionProvider()
      : IExecutionProvider(kType,
                           OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                     OrtDevice::VendorIds::NONE, 0)) {}
};

TEST(ExternalDataLoaderManagerTest, MemcpyTransformerPreservesFileSourceMetadataAndRespectsOuterScopeShadowing) {
  PathString data_path = ORT_TSTR("external_file_source_memcpy_XXXXXX");
  ScopedFileDeleter data_deleter;
  std::string payload(512, '\0');
  ASSERT_NO_FATAL_FAILURE(WriteTestFile(payload, data_path, data_deleter));

  std::unique_ptr<RandomAccessFile> opened_file;
  ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(data_path.c_str(), opened_file));
  std::shared_ptr<RandomAccessFile> shared_file{std::move(opened_file)};
  auto files = std::make_shared<
      std::unordered_map<PathString, std::shared_ptr<RandomAccessFile>>>();
  files->emplace(ORT_TSTR("weights.bin"), shared_file);

  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model_proto.add_opset_import()->set_version(13);
  auto& graph_proto = *model_proto.mutable_graph();
  graph_proto.set_name("memcpy_file_source");

  auto set_float_type = [](ONNX_NAMESPACE::ValueInfoProto* value_info, const char* name, int64_t dim) {
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
  };
  SetBoolType(*graph_proto.add_input(), "cond", true);
  set_float_type(graph_proto.add_output(), "cpu_out", 64);
  set_float_type(graph_proto.add_output(), "shadowed_main_out", 64);
  set_float_type(graph_proto.add_output(), "gpu_out", 64);
  set_float_type(graph_proto.add_output(), "if_out", 64);

  auto add_external_float_tensor = [](ONNX_NAMESPACE::GraphProto& target_graph, const char* name,
                                      int64_t offset, int64_t length) {
    auto* tensor = target_graph.add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor->add_dims(64);
    tensor->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* loc = tensor->add_external_data();
    loc->set_key("location");
    loc->set_value("weights.bin");
    auto* off = tensor->add_external_data();
    off->set_key("offset");
    off->set_value(std::to_string(offset));
    auto* len = tensor->add_external_data();
    len->set_key("length");
    len->set_value(std::to_string(length));
  };

  add_external_float_tensor(graph_proto, "shared_w", 0, 256);
  add_external_float_tensor(graph_proto, "shadowed_w", 256, 256);

  auto* cpu_node = graph_proto.add_node();
  cpu_node->set_name("main_cpu_id");
  cpu_node->set_op_type("Identity");
  cpu_node->add_input("shared_w");
  cpu_node->add_output("cpu_out");

  auto* shadowed_main_node = graph_proto.add_node();
  shadowed_main_node->set_name("main_shadowed_cpu_id");
  shadowed_main_node->set_op_type("Identity");
  shadowed_main_node->add_input("shadowed_w");
  shadowed_main_node->add_output("shadowed_main_out");

  auto* gpu_node = graph_proto.add_node();
  gpu_node->set_name("main_gpu_id");
  gpu_node->set_op_type("Identity");
  gpu_node->add_input("shared_w");
  gpu_node->add_output("gpu_out");

  auto* if_node = graph_proto.add_node();
  if_node->set_name("subgraph_if");
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("if_out");

  // then_branch consumes outer-scope "shared_w" on both CPU and GPU.
  auto* then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto& then_proto = *then_attr->mutable_g();
  then_proto.set_name("then_branch_graph");
  auto* then_cpu = then_proto.add_node();
  then_cpu->set_name("then_cpu_id");
  then_cpu->set_op_type("Identity");
  then_cpu->add_input("shared_w");
  then_cpu->add_output("then_cpu_val");
  auto* then_gpu = then_proto.add_node();
  then_gpu->set_name("then_gpu_id");
  then_gpu->set_op_type("Identity");
  then_gpu->add_input("shared_w");
  then_gpu->add_output("then_gpu_val");
  auto* then_add = then_proto.add_node();
  then_add->set_name("then_add");
  then_add->set_op_type("Add");
  then_add->add_input("then_cpu_val");
  then_add->add_input("then_gpu_val");
  then_add->add_output("then_out");
  set_float_type(then_proto.add_output(), "then_out", 64);

  // else_branch defines a LOCAL non-external initializer named "shadowed_w" that shadows outer-scope "shadowed_w".
  auto* else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  auto& else_proto = *else_attr->mutable_g();
  else_proto.set_name("else_branch_graph");
  auto* local_shadow = else_proto.add_initializer();
  local_shadow->set_name("shadowed_w");
  local_shadow->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  local_shadow->add_dims(64);
  local_shadow->set_raw_data(std::string(256, '\0'));
  auto* else_cpu = else_proto.add_node();
  else_cpu->set_name("else_cpu_id");
  else_cpu->set_op_type("Identity");
  else_cpu->add_input("shadowed_w");
  else_cpu->add_output("else_cpu_val");
  auto* else_gpu = else_proto.add_node();
  else_gpu->set_name("else_gpu_id");
  else_gpu->set_op_type("Identity");
  else_gpu->add_input("shadowed_w");
  else_gpu->add_output("else_gpu_val");
  auto* else_add = else_proto.add_node();
  else_add->set_name("else_add");
  else_add->set_op_type("Add");
  else_add->add_input("else_cpu_val");
  else_add->add_input("else_gpu_val");
  else_add->add_output("else_out");
  set_float_type(else_proto.add_output(), "else_out", 64);

  Model model(model_proto, PathString{}, nullptr, DefaultLoggingManager().DefaultLogger(),
              ModelOptions(true, false, {}, files));
  Graph& graph = model.MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  Node* main_gpu_node = nullptr;
  Graph* then_subgraph = nullptr;
  Graph* else_subgraph = nullptr;
  Node* then_gpu_node = nullptr;
  Node* else_gpu_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.Name() == "main_gpu_id") {
      main_gpu_node = &node;
    } else if (node.Name() == "subgraph_if") {
      node.SetExecutionProviderType(kCpuExecutionProvider);
      then_subgraph = node.GetMutableGraphAttribute("then_branch");
      else_subgraph = node.GetMutableGraphAttribute("else_branch");
    } else {
      node.SetExecutionProviderType(kCpuExecutionProvider);
    }
  }
  ASSERT_NE(main_gpu_node, nullptr);
  ASSERT_NE(then_subgraph, nullptr);
  ASSERT_NE(else_subgraph, nullptr);
  main_gpu_node->SetExecutionProviderType(DummyGpuExecutionProvider::kType);

  for (auto& node : then_subgraph->Nodes()) {
    if (node.Name() == "then_gpu_id") {
      then_gpu_node = &node;
      node.SetExecutionProviderType(DummyGpuExecutionProvider::kType);
    } else {
      node.SetExecutionProviderType(kCpuExecutionProvider);
    }
  }
  for (auto& node : else_subgraph->Nodes()) {
    if (node.Name() == "else_gpu_id") {
      else_gpu_node = &node;
      node.SetExecutionProviderType(DummyGpuExecutionProvider::kType);
    } else {
      node.SetExecutionProviderType(kCpuExecutionProvider);
    }
  }
  ASSERT_NE(then_gpu_node, nullptr);
  ASSERT_NE(else_gpu_node, nullptr);
  else_subgraph->AddOuterScopeNodeArg("shadowed_w");

  ExecutionProviders execution_providers;
  ASSERT_STATUS_OK(execution_providers.Add(DummyGpuExecutionProvider::kType,
                                           std::make_unique<DummyGpuExecutionProvider>()));
  ASSERT_STATUS_OK(execution_providers.Add(kCpuExecutionProvider,
                                           std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo{})));
  KernelRegistryManager registry_manager;
  ASSERT_STATUS_OK(registry_manager.RegisterKernels(execution_providers));
  auto device_registry = std::make_shared<KernelRegistry>();
  KernelDefBuilder device_kernel_def;
  device_kernel_def.SetName("Identity")
      .SetDomain(kOnnxDomain)
      .SinceVersion(1, 13)
      .Provider(DummyGpuExecutionProvider::kType)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  ASSERT_STATUS_OK(device_registry->Register(
      device_kernel_def,
      [](FuncManager&, const OpKernelInfo&, std::unique_ptr<OpKernel>&) { return Status::OK(); }));
  registry_manager.RegisterKernelRegistry(std::move(device_registry));

  InlinedVector<gsl::not_null<const IExecutionProvider*>> provider_ptrs{
      execution_providers.Get(DummyGpuExecutionProvider::kType),
      execution_providers.Get(kCpuExecutionProvider)};
  MemcpyTransformer transformer(provider_ptrs, registry_manager);
  bool modified = false;
  ASSERT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_TRUE(modified);

  // Main graph GPU node must consume a duplicated initializer that preserves shared_w's file source.
  const std::string& main_dup_name = main_gpu_node->InputDefs()[0]->Name();
  EXPECT_NE(main_dup_name, "shared_w");
  Graph::ExternalInitializerFileSource main_dup_source;
  ASSERT_TRUE(graph.GetExternalInitializerFileSource(main_dup_name, main_dup_source));
  EXPECT_EQ(main_dup_source.location, ORT_TSTR("weights.bin"));
  EXPECT_EQ(main_dup_source.file.get(), shared_file.get());
  EXPECT_EQ(main_dup_source.offset, 0);
  EXPECT_EQ(main_dup_source.length, 256u);

  // then_branch GPU node must consume a duplicated initializer that preserves outer-scope shared_w's file source.
  const std::string& then_dup_name = then_gpu_node->InputDefs()[0]->Name();
  EXPECT_NE(then_dup_name, "shared_w");
  Graph::ExternalInitializerFileSource then_dup_source;
  ASSERT_TRUE(then_subgraph->GetExternalInitializerFileSource(then_dup_name, then_dup_source));
  EXPECT_EQ(then_dup_source.location, ORT_TSTR("weights.bin"));
  EXPECT_EQ(then_dup_source.file.get(), shared_file.get());
  EXPECT_EQ(then_dup_source.offset, 0);
  EXPECT_EQ(then_dup_source.length, 256u);

  // else_branch's local "shadowed_w" shadows outer-scope "shadowed_w" (which remains live on the main graph),
  // so neither "shadowed_w" nor its GPU duplicate may inherit the outer-scope file source.
  EXPECT_TRUE(graph.HasExternalInitializerFileSource("shadowed_w", /*check_outer_scope=*/false));
  EXPECT_FALSE(else_subgraph->HasExternalInitializerFileSource("shadowed_w", /*check_outer_scope=*/false));
  EXPECT_FALSE(else_subgraph->HasExternalInitializerFileSource("shadowed_w", /*check_outer_scope=*/true));
  const std::string& else_dup_name = else_gpu_node->InputDefs()[0]->Name();
  EXPECT_NE(else_dup_name, "shadowed_w");
  EXPECT_FALSE(else_subgraph->HasExternalInitializerFileSource(else_dup_name, /*check_outer_scope=*/false));
  EXPECT_FALSE(else_subgraph->HasExternalInitializerFileSource(else_dup_name, /*check_outer_scope=*/true));
}

#ifdef _WIN32
TEST(ExternalDataLoaderManagerTest, DuplicateReadOnlyFileHandleReadPreservesCallerCursor) {
  PathString data_path = ORT_TSTR("external_file_source_cursor_XXXXXX");
  ScopedFileDeleter data_deleter;
  std::string payload = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  ASSERT_NO_FATAL_FAILURE(WriteTestFile(payload, data_path, data_deleter));

  const HANDLE raw_handle = CreateFileW(
      data_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
      nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(raw_handle, INVALID_HANDLE_VALUE);
  std::unique_ptr<void, decltype(&CloseHandle)> caller_file(raw_handle, CloseHandle);

  LARGE_INTEGER requested_pos{};
  requested_pos.QuadPart = 7;
  LARGE_INTEGER new_pos{};
  ASSERT_TRUE(SetFilePointerEx(raw_handle, requested_pos, &new_pos, FILE_BEGIN));
  ASSERT_EQ(new_pos.QuadPart, 7);

  std::shared_ptr<RandomAccessFile> duplicated;
  ASSERT_STATUS_OK(Env::Default().DuplicateReadOnlyFileHandle(raw_handle, duplicated));
  ASSERT_NE(duplicated, nullptr);

  std::array<char, 10> buffer{};
  ASSERT_STATUS_OK(duplicated->Read(16, gsl::make_span(buffer.data(), buffer.size())));
  EXPECT_EQ(std::string_view(buffer.data(), buffer.size()),
            std::string_view(payload.data() + 16, buffer.size()));

  // Reading past EOF must still fail with unexpected end of file.
  EXPECT_FALSE(duplicated->Read(static_cast<FileOffsetType>(payload.size() - 4),
                                gsl::make_span(buffer.data(), buffer.size()))
                   .IsOK());

  // The caller's synchronous file handle cursor must remain at 7.
  LARGE_INTEGER current_pos{};
  const LARGE_INTEGER zero{};
  ASSERT_TRUE(SetFilePointerEx(raw_handle, zero, &current_pos, FILE_CURRENT));
  EXPECT_EQ(current_pos.QuadPart, 7);
}
#endif
#endif

#if !defined(ORT_MINIMAL_BUILD)
class BatchLifecycleExternalDataLoader final : public IExternalDataLoader {
 public:
  enum class FailurePoint {
    None,
    Begin,
    CommitCandidates,
    Finalize,
  };

  explicit BatchLifecycleExternalDataLoader(FailurePoint failure_point = FailurePoint::None)
      : failure_point_{failure_point} {
  }

  bool CanLoad(const OrtMemoryInfo&) const override {
    return false;
  }

  common::Status BeginLoad() const override {
    ++begin_count;
    return failure_point_ == FailurePoint::Begin
               ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "begin failure")
               : common::Status::OK();
  }

  common::Status FinalizeLoad(const std::function<bool()>&) const override {
    ++finalize_count;
    return failure_point_ == FailurePoint::Finalize
               ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "finalize failure")
               : common::Status::OK();
  }

#if defined(ENABLE_D3D12_FILE_LOADING)
  common::Status CommitLoadCandidates(const std::function<bool()>&) const override {
    ++commit_candidates_count;
    return failure_point_ == FailurePoint::CommitCandidates
               ? ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "commit failure")
               : common::Status::OK();
  }
#endif

  void EndLoad() const noexcept override {
    ++end_count;
  }

  mutable int begin_count = 0;
  mutable int commit_candidates_count = 0;
  mutable int finalize_count = 0;
  mutable int end_count = 0;

 private:
  FailurePoint failure_point_;
};

TEST(ExternalDataLoaderManagerTest, DefaultBatchLifecycleIsBackwardCompatible) {
  class LegacyExternalDataLoader final : public IExternalDataLoader {
   public:
    bool CanLoad(const OrtMemoryInfo&) const override {
      return false;
    }
  };

  ExternalDataLoaderManager manager;
  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::make_unique<LegacyExternalDataLoader>()));
  EXPECT_STATUS_OK(manager.BeginLoad());
#if defined(ENABLE_D3D12_FILE_LOADING)
  EXPECT_STATUS_OK(manager.CommitLoadCandidates([]() { return false; }));
#endif
  EXPECT_STATUS_OK(manager.FinalizeLoad([]() { return false; }));
  manager.EndLoad();
}

TEST(ExternalDataLoaderManagerTest, BeginFailureEndsEveryLoader) {
  ExternalDataLoaderManager manager;
  auto first = std::make_unique<BatchLifecycleExternalDataLoader>();
  auto* first_ptr = first.get();
  auto failing = std::make_unique<BatchLifecycleExternalDataLoader>(
      BatchLifecycleExternalDataLoader::FailurePoint::Begin);
  auto* failing_ptr = failing.get();

  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(first)));
  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(failing)));

  EXPECT_FALSE(manager.BeginLoad().IsOK());
  EXPECT_EQ(first_ptr->begin_count, 1);
  EXPECT_EQ(failing_ptr->begin_count, 1);
  EXPECT_EQ(first_ptr->end_count, 1);
  EXPECT_EQ(failing_ptr->end_count, 1);
}

TEST(ExternalDataLoaderManagerTest, FinalizeFailureEndsEveryLoader) {
  ExternalDataLoaderManager manager;
  auto failing = std::make_unique<BatchLifecycleExternalDataLoader>(
      BatchLifecycleExternalDataLoader::FailurePoint::Finalize);
  auto* failing_ptr = failing.get();
  auto second = std::make_unique<BatchLifecycleExternalDataLoader>();
  auto* second_ptr = second.get();

  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(failing)));
  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(second)));

  ASSERT_STATUS_OK(manager.BeginLoad());
  EXPECT_FALSE(manager.FinalizeLoad([]() { return false; }).IsOK());
  EXPECT_EQ(failing_ptr->finalize_count, 1);
  EXPECT_EQ(second_ptr->finalize_count, 0);
  EXPECT_EQ(failing_ptr->end_count, 1);
  EXPECT_EQ(second_ptr->end_count, 1);
}

#if defined(ENABLE_D3D12_FILE_LOADING)
TEST(ExternalDataLoaderManagerTest, CommitCandidatesFailureEndsEveryLoader) {
  ExternalDataLoaderManager manager;
  auto failing = std::make_unique<BatchLifecycleExternalDataLoader>(
      BatchLifecycleExternalDataLoader::FailurePoint::CommitCandidates);
  auto* failing_ptr = failing.get();
  auto uncommitted = std::make_unique<BatchLifecycleExternalDataLoader>();
  auto* uncommitted_ptr = uncommitted.get();

  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(failing)));
  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(uncommitted)));

  ASSERT_STATUS_OK(manager.BeginLoad());
  EXPECT_FALSE(manager.CommitLoadCandidates([]() { return false; }).IsOK());
  EXPECT_EQ(failing_ptr->commit_candidates_count, 1);
  EXPECT_EQ(uncommitted_ptr->commit_candidates_count, 0);
  EXPECT_EQ(failing_ptr->end_count, 1);
  EXPECT_EQ(uncommitted_ptr->end_count, 1);
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
struct NativeCreatorRecord {
  PathString location;
  const RandomAccessFile* file = nullptr;
  std::string tensor_name;
  FileOffsetType offset = 0;
  size_t length = 0;
};

struct NativeCreatorState {
  std::vector<NativeCreatorRecord> registered_candidates;
  std::vector<NativeCreatorRecord> loaded_tensors;
  int commit_count = 0;
};

class NativeSourceTrackingCreatorLoader final : public IExternalDataLoader {
 public:
  explicit NativeSourceTrackingCreatorLoader(std::shared_ptr<NativeCreatorState> state)
      : state_(std::move(state)) {}

  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override {
    return target_memory_info.device.Type() == OrtDevice::CPU;
  }

  bool CreatesTensorForDevice(const OrtDevice& target_device) const override {
    return target_device.Type() == OrtDevice::CPU;
  }

  common::Status RegisterLoadCandidate(const PathString& data_file_name,
                                       const RandomAccessFile& file,
                                       std::string_view tensor_name,
                                       FileOffsetType data_offset,
                                       SafeInt<size_t> data_length) const override {
    state_->registered_candidates.push_back(
        NativeCreatorRecord{data_file_name, &file, std::string(tensor_name), data_offset,
                            static_cast<size_t>(data_length)});
    return common::Status::OK();
  }

  common::Status CommitLoadCandidates(const std::function<bool()>&) const override {
    ++state_->commit_count;
    return common::Status::OK();
  }

  common::Status LoadTensor(const PathString& data_file_name,
                            const RandomAccessFile& file,
                            std::string_view tensor_name,
                            FileOffsetType data_offset,
                            SafeInt<size_t> data_length,
                            const std::shared_ptr<IAllocator>& allocator,
                            Tensor& tensor) const override {
    state_->loaded_tensors.push_back(
        NativeCreatorRecord{data_file_name, &file, std::string(tensor_name), data_offset,
                            static_cast<size_t>(data_length)});
    Tensor allocated(tensor.DataType(), tensor.Shape(), allocator);
    ORT_RETURN_IF_ERROR(file.Read(
        data_offset,
        gsl::span<char>(static_cast<char*>(allocated.MutableDataRaw()), allocated.SizeInBytes())));
    tensor = std::move(allocated);
    return common::Status::OK();
  }

 private:
  std::shared_ptr<NativeCreatorState> state_;
};

class CPUExecutionProviderWithNativeCreatorLoader final : public CPUExecutionProvider {
 public:
  explicit CPUExecutionProviderWithNativeCreatorLoader(std::shared_ptr<NativeCreatorState> state)
      : CPUExecutionProvider(CPUExecutionProviderInfo{}), state_(std::move(state)) {}

  std::unique_ptr<IExternalDataLoader> GetExternalDataLoader() const override {
    return std::make_unique<NativeSourceTrackingCreatorLoader>(state_);
  }

 private:
  std::shared_ptr<NativeCreatorState> state_;
};

TEST(ExternalDataLoaderManagerTest, NativeSourceCreatorInvokedAfterConstantIfInlining) {
  PathString data_path = ORT_TSTR("external_native_creator_inline_if_XXXXXX");
  ScopedFileDeleter data_deleter;
  std::vector<float> weights(64, 3.0f);
  std::string payload(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
  ASSERT_NO_FATAL_FAILURE(WriteTestFile(payload, data_path, data_deleter));

  std::unique_ptr<RandomAccessFile> opened_file;
  ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(data_path.c_str(), opened_file));
  std::shared_ptr<RandomAccessFile> shared_file{std::move(opened_file)};
  auto files = std::make_shared<
      std::unordered_map<PathString, std::shared_ptr<RandomAccessFile>>>();
  files->emplace(ORT_TSTR("virtual/weights.bin"), shared_file);

  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model_proto.add_opset_import()->set_version(13);
  auto& graph_proto = *model_proto.mutable_graph();
  graph_proto.set_name("native_creator_inline_if");

  auto* cond_init = graph_proto.add_initializer();
  cond_init->set_name("cond");
  cond_init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  cond_init->add_int32_data(1);

  auto set_float_type = [](ONNX_NAMESPACE::ValueInfoProto* value_info, const char* name) {
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(64);
  };
  set_float_type(graph_proto.add_input(), "X");
  set_float_type(graph_proto.add_output(), "Y");

  auto* if_node = graph_proto.add_node();
  if_node->set_op_type("If");
  if_node->set_name("const_if");
  if_node->add_input("cond");
  if_node->add_output("Y");

  for (const char* branch_name : {"then_branch", "else_branch"}) {
    auto* attr = if_node->add_attribute();
    attr->set_name(branch_name);
    attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
    auto& branch = *attr->mutable_g();
    branch.set_name(branch_name);
    auto* weight = branch.add_initializer();
    weight->set_name("branch_w");
    weight->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    weight->add_dims(64);
    weight->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* loc = weight->add_external_data();
    loc->set_key("location");
    loc->set_value("virtual/weights.bin");
    auto* off = weight->add_external_data();
    off->set_key("offset");
    off->set_value("0");
    auto* len = weight->add_external_data();
    len->set_key("length");
    len->set_value("256");
    auto* add = branch.add_node();
    add->set_op_type("Add");
    add->add_input("X");
    add->add_input("branch_w");
    add->add_output("branch_out");
    set_float_type(branch.add_output(), "branch_out");
  }

  std::string serialized = model_proto.SerializeAsString();
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Level1;
  options.external_initializer_file_sources = files;

  auto state = std::make_shared<NativeCreatorState>();
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(
      std::make_unique<CPUExecutionProviderWithNativeCreatorLoader>(state)));
  ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
  ASSERT_STATUS_OK(session.Initialize());

  ASSERT_EQ(state->registered_candidates.size(), 1u);
  EXPECT_EQ(state->registered_candidates[0].location, ORT_TSTR("virtual/weights.bin"));
  EXPECT_EQ(state->registered_candidates[0].file, shared_file.get());
  EXPECT_EQ(state->registered_candidates[0].offset, 0);
  EXPECT_EQ(state->registered_candidates[0].length, 256u);
  EXPECT_EQ(state->commit_count, 1);
  ASSERT_EQ(state->loaded_tensors.size(), 1u);
  EXPECT_EQ(state->loaded_tensors[0].tensor_name, state->registered_candidates[0].tensor_name);
  EXPECT_EQ(state->loaded_tensors[0].file, shared_file.get());
}

class GpuExecutionProviderWithNativeCreatorLoader final : public IExecutionProvider {
 public:
  static constexpr const char* kType = "GpuNativeCreatorTestExecutionProvider";

  explicit GpuExecutionProviderWithNativeCreatorLoader(std::shared_ptr<NativeCreatorState> state)
      : IExecutionProvider(kType,
                           OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                     OrtDevice::VendorIds::NONE, 0)),
        state_(std::move(state)) {}

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    class NoOpKernel final : public OpKernel {
     public:
      explicit NoOpKernel(const OpKernelInfo& info) : OpKernel(info) {}
      Status Compute(OpKernelContext*) const override { return Status::OK(); }
    };
    auto create_kernel = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
      out = std::make_unique<NoOpKernel>(info);
      return Status::OK();
    };
    auto registry = std::make_shared<KernelRegistry>();
    for (const char* op_name : {"Add", "MemcpyFromHost", "MemcpyToHost"}) {
      KernelDefBuilder def;
      def.SetName(op_name)
          .SetDomain(kOnnxDomain)
          .SinceVersion(1, 13)
          .Provider(kType)
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
      ORT_THROW_IF_ERROR(registry->Register(def, create_kernel));
    }
    return registry;
  }

  std::unique_ptr<IExternalDataLoader> GetExternalDataLoader() const override {
    class GpuCreatorLoader final : public IExternalDataLoader {
     public:
      explicit GpuCreatorLoader(std::shared_ptr<NativeCreatorState> state) : state_(std::move(state)) {}
      bool CanLoad(const OrtMemoryInfo& target_memory_info) const override {
        return target_memory_info.device.Type() == OrtDevice::GPU;
      }
      bool CreatesTensorForDevice(const OrtDevice& target_device) const override {
        return target_device.Type() == OrtDevice::GPU;
      }
      common::Status RegisterLoadCandidate(const PathString& data_file_name,
                                           const RandomAccessFile& file,
                                           std::string_view tensor_name,
                                           FileOffsetType data_offset,
                                           SafeInt<size_t> data_length) const override {
        state_->registered_candidates.push_back(
            NativeCreatorRecord{data_file_name, &file, std::string(tensor_name), data_offset,
                                static_cast<size_t>(data_length)});
        return common::Status::OK();
      }
      common::Status CommitLoadCandidates(const std::function<bool()>&) const override {
        ++state_->commit_count;
        return common::Status::OK();
      }
      common::Status LoadTensor(const PathString& data_file_name,
                                const RandomAccessFile& file,
                                std::string_view tensor_name,
                                FileOffsetType data_offset,
                                SafeInt<size_t> data_length,
                                const std::shared_ptr<IAllocator>& allocator,
                                Tensor& tensor) const override {
        state_->loaded_tensors.push_back(
            NativeCreatorRecord{data_file_name, &file, std::string(tensor_name), data_offset,
                                static_cast<size_t>(data_length)});
        Tensor allocated(tensor.DataType(), tensor.Shape(), allocator);
        ORT_RETURN_IF_ERROR(file.Read(
            data_offset,
            gsl::span<char>(static_cast<char*>(allocated.MutableDataRaw()), allocated.SizeInBytes())));
        tensor = std::move(allocated);
        return common::Status::OK();
      }

     private:
      std::shared_ptr<NativeCreatorState> state_;
    };
    return std::make_unique<GpuCreatorLoader>(state_);
  }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override {
    return {std::make_shared<CPUAllocator>(
        OrtMemoryInfo("GpuNativeCreatorTest", OrtAllocatorType::OrtDeviceAllocator,
                      OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                OrtDevice::VendorIds::NONE, 0)))};
  }

  std::unique_ptr<IDataTransfer> GetDataTransfer() const override {
    class LoopbackGpuDataTransfer final : public IDataTransfer {
     public:
      bool CanCopy(const OrtDevice& src, const OrtDevice& dst) const override {
        return src.Type() == OrtDevice::GPU || dst.Type() == OrtDevice::GPU;
      }
      common::Status CopyTensor(const Tensor& src, Tensor& dst) const override {
        std::memcpy(dst.MutableDataRaw(), src.DataRaw(), src.SizeInBytes());
        return common::Status::OK();
      }
    };
    return std::make_unique<LoopbackGpuDataTransfer>();
  }

 private:
  std::shared_ptr<NativeCreatorState> state_;
};

TEST(ExternalDataLoaderManagerTest, NativeSourceCreatorInvokedAfterMixedProviderMemcpyDuplication) {
  PathString data_path = ORT_TSTR("external_native_creator_memcpy_XXXXXX");
  ScopedFileDeleter data_deleter;
  std::vector<float> weights(64, 5.0f);
  std::string payload(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
  ASSERT_NO_FATAL_FAILURE(WriteTestFile(payload, data_path, data_deleter));

  std::unique_ptr<RandomAccessFile> opened_file;
  ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(data_path.c_str(), opened_file));
  std::shared_ptr<RandomAccessFile> shared_file{std::move(opened_file)};
  auto files = std::make_shared<
      std::unordered_map<PathString, std::shared_ptr<RandomAccessFile>>>();
  files->emplace(ORT_TSTR("virtual/weights.bin"), shared_file);

  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model_proto.add_opset_import()->set_version(13);
  auto& graph_proto = *model_proto.mutable_graph();
  graph_proto.set_name("native_creator_memcpy");

  auto set_float_type = [](ONNX_NAMESPACE::ValueInfoProto* value_info, const char* name) {
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(64);
  };
  set_float_type(graph_proto.add_input(), "X");
  set_float_type(graph_proto.add_output(), "Y_gpu");
  set_float_type(graph_proto.add_output(), "Y_cpu");

  auto* weight = graph_proto.add_initializer();
  weight->set_name("shared_w");
  weight->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  weight->add_dims(64);
  weight->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* loc = weight->add_external_data();
  loc->set_key("location");
  loc->set_value("virtual/weights.bin");
  auto* off = weight->add_external_data();
  off->set_key("offset");
  off->set_value("0");
  auto* len = weight->add_external_data();
  len->set_key("length");
  len->set_value("256");

  // Add is claimed by GpuExecutionProviderWithNativeCreatorLoader; Sub remains on CPUExecutionProvider.
  // Both consume "shared_w", so MemcpyTransformer duplicates "shared_w" for the GPU provider.
  auto* gpu_add = graph_proto.add_node();
  gpu_add->set_name("gpu_add");
  gpu_add->set_op_type("Add");
  gpu_add->add_input("X");
  gpu_add->add_input("shared_w");
  gpu_add->add_output("Y_gpu");

  auto* cpu_sub = graph_proto.add_node();
  cpu_sub->set_name("cpu_sub");
  cpu_sub->set_op_type("Sub");
  cpu_sub->add_input("X");
  cpu_sub->add_input("shared_w");
  cpu_sub->add_output("Y_cpu");

  std::string serialized = model_proto.SerializeAsString();
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.external_initializer_file_sources = files;

  auto state = std::make_shared<NativeCreatorState>();
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(
      std::make_unique<GpuExecutionProviderWithNativeCreatorLoader>(state)));
  ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
  ASSERT_STATUS_OK(session.Initialize());

  ASSERT_EQ(state->registered_candidates.size(), 1u);
  EXPECT_NE(state->registered_candidates[0].tensor_name, "shared_w");
  EXPECT_EQ(state->registered_candidates[0].location, ORT_TSTR("virtual/weights.bin"));
  EXPECT_EQ(state->registered_candidates[0].file, shared_file.get());
  EXPECT_EQ(state->registered_candidates[0].offset, 0);
  EXPECT_EQ(state->registered_candidates[0].length, 256u);
  EXPECT_EQ(state->commit_count, 1);
  ASSERT_EQ(state->loaded_tensors.size(), 1u);
  EXPECT_EQ(state->loaded_tensors[0].tensor_name, state->registered_candidates[0].tensor_name);
  EXPECT_EQ(state->loaded_tensors[0].file, shared_file.get());
}
#endif
#endif
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) && !defined(__wasm__) && !defined(ORT_NO_EXCEPTIONS)
class ThrowingReadFile final : public RandomAccessFile {
 public:
  explicit ThrowingReadFile(size_t successful_reads) : successful_reads_{successful_reads} {}

  common::Status GetLength(uint64_t& length) const override {
    length = std::numeric_limits<uint64_t>::max();
    return common::Status::OK();
  }

  common::Status GetCanonicalPath(PathString&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "not supported");
  }

  common::Status Map(FileOffsetType, size_t, MappedMemoryPtr&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "not supported");
  }

  common::Status Read(FileOffsetType, gsl::span<char> buffer) const override {
    if (reads_.fetch_add(1) >= successful_reads_) {
      throw std::runtime_error("simulated read exception");
    }
    std::fill(buffer.begin(), buffer.end(), 0);
    return common::Status::OK();
  }

 private:
  size_t successful_reads_;
  mutable std::atomic<size_t> reads_{0};
};

// A throwing read after a successful async copy submission must still drain every used stream.
void VerifyStreamsDrainedAfterThrowingRead(size_t reading_threads, size_t total_bytes, size_t single_buffer_size,
                                           size_t successful_reads) {
  constexpr size_t kMaxStreams = 16;
  std::array<std::atomic<bool>, kMaxStreams> in_flight{};
  std::atomic<size_t> copies_submitted{0};

  std::vector<uint8_t> staging0(single_buffer_size);
  std::vector<uint8_t> staging1(single_buffer_size);
  const std::array<void*, 2> buffers{staging0.data(), staging1.data()};
  std::vector<uint8_t> destination(total_bytes);

  ThrowingReadFile file(successful_reads);
  const ExternalDataLoadTask task{&file, 0, destination.data(), total_bytes, 1, false};
  std::unique_ptr<ExternalDataLoaderThreadPool> reader_pool;

  const auto status = ExecuteExternalDataLoadTasks(
      gsl::span<const ExternalDataLoadTask>(&task, 1),
      /*allow_small_parallel=*/false,
      reading_threads,
      reading_threads,
      single_buffer_size,
      /*max_chunk_bytes=*/single_buffer_size,
      buffers,
      reader_pool,
      std::function<bool()>{},
      [](size_t) { return common::Status::OK(); },
      [&](size_t stream_index) {
        in_flight[stream_index] = false;
        return common::Status::OK();
      },
      [&](void*, void*, size_t, size_t stream_index) {
        in_flight[stream_index] = true;
        ++copies_submitted;
        return common::Status::OK();
      });

  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("simulated read exception"), std::string::npos) << status.ErrorMessage();
  EXPECT_GT(copies_submitted.load(), 0U);
  for (size_t i = 0; i < kMaxStreams; ++i) {
    EXPECT_FALSE(in_flight[i].load()) << "stream " << i << " was not drained after the read exception";
  }
}

class PatternFile final : public RandomAccessFile {
 public:
  common::Status GetLength(uint64_t& length) const override {
    length = std::numeric_limits<uint64_t>::max();
    return common::Status::OK();
  }

  common::Status GetCanonicalPath(PathString&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "not supported");
  }

  common::Status Map(FileOffsetType, size_t, MappedMemoryPtr&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "not supported");
  }

  common::Status Read(FileOffsetType offset, gsl::span<char> buffer) const override {
    for (size_t i = 0; i < buffer.size(); ++i) {
      buffer[i] = static_cast<char>((static_cast<size_t>(offset) + i) * 31 + 7);
    }
    return common::Status::OK();
  }
};

TEST(ExternalDataLoadTasksTest, ParallelPathCoversEveryByteOfManyUnevenTasks) {
  constexpr size_t kSingleBufferSize = 64 * 1024;
  constexpr size_t kThreads = 3;
  const std::vector<size_t> lengths{5000, 0, 3 * 4096 + 17, 1, 1024 * 1024 + 1, 16384, 40000};

  PatternFile file;
  std::vector<std::vector<uint8_t>> destinations;
  std::vector<ExternalDataLoadTask> tasks;
  destinations.reserve(lengths.size());
  FileOffsetType file_offset = 13;
  for (const size_t length : lengths) {
    destinations.emplace_back(length, uint8_t{0});
    tasks.push_back(ExternalDataLoadTask{&file, file_offset, destinations.back().data(), length, 1, false});
    file_offset += static_cast<FileOffsetType>(length) + 5;
  }

  std::vector<uint8_t> staging0(kSingleBufferSize);
  std::vector<uint8_t> staging1(kSingleBufferSize);
  const std::array<void*, 2> buffers{staging0.data(), staging1.data()};
  std::unique_ptr<ExternalDataLoaderThreadPool> reader_pool;

  ASSERT_STATUS_OK(ExecuteExternalDataLoadTasks(
      gsl::span<const ExternalDataLoadTask>(tasks),
      /*allow_small_parallel=*/true, kThreads, kThreads, kSingleBufferSize,
      /*max_chunk_bytes=*/16384, buffers, reader_pool, std::function<bool()>{},
      [](size_t) { return common::Status::OK(); },
      [](size_t) { return common::Status::OK(); },
      [](void* dst, void* src, size_t bytes, size_t) {
        std::memcpy(dst, src, bytes);
        return common::Status::OK();
      }));

  ASSERT_NE(reader_pool, nullptr);
  for (size_t t = 0; t < tasks.size(); ++t) {
    for (size_t i = 0; i < lengths[t]; ++i) {
      const auto expected = static_cast<uint8_t>((static_cast<size_t>(tasks[t].file_offset) + i) * 31 + 7);
      ASSERT_EQ(destinations[t][i], expected) << "task " << t << " byte " << i;
    }
  }
}

TEST(ExternalDataLoadTasksTest, SerialPathDrainsStreamsWhenReadThrows) {
  VerifyStreamsDrainedAfterThrowingRead(/*reading_threads=*/1, /*total_bytes=*/3 * 1024 * 1024,
                                        /*single_buffer_size=*/1024 * 1024, /*successful_reads=*/1);
}

TEST(ExternalDataLoadTasksTest, ParallelPathDrainsStreamsWhenReadThrows) {
  VerifyStreamsDrainedAfterThrowingRead(/*reading_threads=*/2, /*total_bytes=*/16 * 1024 * 1024,
                                        /*single_buffer_size=*/1024 * 1024, /*successful_reads=*/3);
}
#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime
