// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader_manager.h"

#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test/util/include/temp_dir.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {
namespace {

class BatchLifecycleExternalDataLoader final : public IExternalDataLoader {
 public:
  enum class FailurePoint {
    None,
    Begin,
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

  void AbortLoad() const noexcept override {
    ++abort_count;
  }

  mutable int begin_count = 0;
  mutable int finalize_count = 0;
  mutable int abort_count = 0;

 private:
  FailurePoint failure_point_;
};

class PreloadTrackingExternalDataLoader final : public IExternalDataLoader {
 public:
  bool CanLoad(const OrtMemoryInfo&) const override {
    return false;
  }

  bool SupportsPreload() const override {
    return true;
  }

  common::Status PreloadTensor(
      const Env&,
      const std::filesystem::path&,
      std::string_view tensor_name,
      FileOffsetType,
      SafeInt<size_t>) const override {
    preloaded_tensor_names.emplace_back(tensor_name);
    return common::Status::OK();
  }

  mutable std::vector<std::string> preloaded_tensor_names;
};

ONNX_NAMESPACE::TensorProto CreateExternalTensorProto(
    const std::string& name, const std::string& location) {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.add_dims(1);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* location_entry = tensor_proto.add_external_data();
  location_entry->set_key("location");
  location_entry->set_value(location);
  auto* offset_entry = tensor_proto.add_external_data();
  offset_entry->set_key("offset");
  offset_entry->set_value("0");
  auto* length_entry = tensor_proto.add_external_data();
  length_entry->set_key("length");
  length_entry->set_value(std::to_string(sizeof(float)));
  return tensor_proto;
}

TEST(ExternalDataLoaderManagerTest, DefaultBatchLifecycleIsBackwardCompatible) {
  class LegacyExternalDataLoader final : public IExternalDataLoader {
   public:
    bool CanLoad(const OrtMemoryInfo&) const override {
      return false;
    }
  };

  ExternalDataLoaderManager manager;
  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::make_unique<LegacyExternalDataLoader>()));
  EXPECT_FALSE(manager.HasPreloader());
  EXPECT_STATUS_OK(manager.BeginLoad());
  EXPECT_STATUS_OK(manager.FinalizeLoad([]() { return false; }));
  manager.AbortLoad();
}

TEST(ExternalDataLoaderManagerTest, BeginFailureAbortsEveryLoader) {
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
  EXPECT_EQ(first_ptr->abort_count, 1);
  EXPECT_EQ(failing_ptr->abort_count, 1);
}

TEST(ExternalDataLoaderManagerTest, FinalizeFailureAbortsEveryLoader) {
  ExternalDataLoaderManager manager;
  auto failing = std::make_unique<BatchLifecycleExternalDataLoader>(
      BatchLifecycleExternalDataLoader::FailurePoint::Finalize);
  auto* failing_ptr = failing.get();
  auto unfinalized = std::make_unique<BatchLifecycleExternalDataLoader>();
  auto* unfinalized_ptr = unfinalized.get();

  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(failing)));
  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(unfinalized)));

  ASSERT_STATUS_OK(manager.BeginLoad());
  EXPECT_FALSE(manager.FinalizeLoad([]() { return false; }).IsOK());
  EXPECT_EQ(failing_ptr->finalize_count, 1);
  EXPECT_EQ(unfinalized_ptr->finalize_count, 0);
  EXPECT_EQ(failing_ptr->abort_count, 1);
  EXPECT_EQ(unfinalized_ptr->abort_count, 1);
}

TEST(ExternalDataLoaderManagerTest, PreloadSkipsExcludedInitializers) {
  TemporaryDirectory temp_dir{ORT_TSTR("external_data_preload_exclusions")};
  const auto temp_path = std::filesystem::path{temp_dir.Path()};
  const auto data_path = temp_path / ORT_TSTR("included.bin");
  {
    std::ofstream stream{data_path, std::ios::binary | std::ios::trunc};
    const float value = 1.0f;
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    ASSERT_TRUE(stream.good());
  }

  Model model{"external_data_preload_exclusions", false,
              DefaultLoggingManager().DefaultLogger()};
  Graph& graph = model.MainGraph();
  graph.AddInitializedTensor(
      CreateExternalTensorProto("included", "included.bin"));
  graph.AddInitializedTensor(
      CreateExternalTensorProto("excluded", "missing.bin"));

  ExternalDataLoaderManager manager;
  auto loader = std::make_unique<PreloadTrackingExternalDataLoader>();
  auto* loader_ptr = loader.get();
  ASSERT_STATUS_OK(manager.RegisterExternalDataLoader(std::move(loader)));

  const std::unordered_set<std::string> excluded_initializer_names{"excluded"};
  ASSERT_STATUS_OK(manager.PreloadExternalData(
      Env::Default(), temp_path / ORT_TSTR("model.onnx"), graph,
      excluded_initializer_names, []() { return false; }));
  EXPECT_EQ(loader_ptr->preloaded_tensor_names,
            std::vector<std::string>{"included"});
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
