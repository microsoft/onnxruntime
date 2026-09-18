// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

#include "core/common/inlined_containers.h"
#include "core/framework/external_data_loader.h"
#include "core/framework/session_state.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"

namespace onnxruntime {
namespace test {
namespace {

enum class ReadFailure { None,
                         Status,
                         Exception };

struct LoaderState {
  size_t created{0};
  size_t destroyed{0};
  bool fail_creation{false};
  ReadFailure failure{ReadFailure::None};
  InlinedVector<FileOffsetType> offsets;
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

  Status LoadTensor(const Env& env, const std::filesystem::path& path, FileOffsetType offset,
                    SafeInt<size_t> length, Tensor& tensor) const override {
    state_->offsets.push_back(offset);
    if (state_->failure == ReadFailure::Exception) {
      ORT_THROW("external loader read exception");
    }
    ORT_RETURN_IF(state_->failure == ReadFailure::Status, "external loader read failure");
    return env.ReadFileIntoBuffer(path.c_str(), offset, length,
                                  gsl::span<char>(static_cast<char*>(tensor.MutableDataRaw()), tensor.SizeInBytes()));
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

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
