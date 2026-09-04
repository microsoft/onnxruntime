// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader_manager.h"

#include <memory>

#include "gtest/gtest.h"

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

}  // namespace
}  // namespace test
}  // namespace onnxruntime
