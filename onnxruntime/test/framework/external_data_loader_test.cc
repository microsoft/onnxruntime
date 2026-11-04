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
#include "core/framework/external_data_loader.h"
#include "core/framework/session_state.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
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
