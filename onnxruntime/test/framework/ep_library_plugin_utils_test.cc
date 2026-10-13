// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_library_plugin_utils.h"

#include "gtest/gtest.h"

#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime::test {

namespace {

size_t released_factory_count = 0;

OrtEpFactory* CreateTestFactory() {
  return new OrtEpFactory{};
}

OrtStatus* ReleaseTestFactory(OrtEpFactory* factory) {
  ++released_factory_count;
  delete factory;
  return nullptr;
}

OrtStatus* CreateOneFactory(const char*, const OrtApiBase*, const OrtLogger*,
                            OrtEpFactory** factories, size_t, size_t* num_factories) {
  factories[0] = CreateTestFactory();
  *num_factories = 1;
  return nullptr;
}

OrtStatus* CreateOneFactoryThenFail(const char*, const OrtApiBase*, const OrtLogger*,
                                    OrtEpFactory** factories, size_t, size_t* num_factories) {
  factories[0] = CreateTestFactory();
  *num_factories = 1;
  return Ort::GetApi().CreateStatus(ORT_FAIL, "injected factory creation failure");
}

OrtStatus* ReturnOversizedFactoryCount(const char*, const OrtApiBase*, const OrtLogger*,
                                       OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  factories[0] = CreateTestFactory();
  *num_factories = max_factories + 1;
  return nullptr;
}

OrtStatus* ReturnInconsistentFactoryOutputs(const char*, const OrtApiBase*, const OrtLogger*,
                                            OrtEpFactory** factories, size_t, size_t* num_factories) {
  factories[1] = CreateTestFactory();
  *num_factories = 1;
  return nullptr;
}

}  // namespace

class EpLibraryPluginUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    released_factory_count = 0;
  }
};

TEST_F(EpLibraryPluginUtilsTest, CreateFactoriesAdoptsFactoriesOnSuccess) {
  std::vector<ep_library_plugin_utils::OrtEpFactoryUniquePtr> factories;

  ASSERT_TRUE(ep_library_plugin_utils::CreateFactories(
                  CreateOneFactory, ReleaseTestFactory, "test plugin", factories)
                  .IsOK());
  ASSERT_EQ(factories.size(), 1u);
  EXPECT_EQ(released_factory_count, 0u);

  factories.clear();
  EXPECT_TRUE(factories.empty());
  EXPECT_EQ(released_factory_count, 1u);
}

TEST_F(EpLibraryPluginUtilsTest, CreateFactoriesReleasesFactoryWhenCreationFails) {
  std::vector<ep_library_plugin_utils::OrtEpFactoryUniquePtr> factories;

  const auto status = ep_library_plugin_utils::CreateFactories(
      CreateOneFactoryThenFail, ReleaseTestFactory, "test plugin", factories);

  EXPECT_FALSE(status.IsOK());
  EXPECT_TRUE(factories.empty());
  EXPECT_EQ(released_factory_count, 1u);
}

TEST_F(EpLibraryPluginUtilsTest, CreateFactoriesRejectsOversizedFactoryCount) {
  std::vector<ep_library_plugin_utils::OrtEpFactoryUniquePtr> factories;

  const auto status = ep_library_plugin_utils::CreateFactories(
      ReturnOversizedFactoryCount, ReleaseTestFactory, "test plugin", factories);

  EXPECT_FALSE(status.IsOK());
  EXPECT_TRUE(factories.empty());
  EXPECT_EQ(released_factory_count, 1u);
}

TEST_F(EpLibraryPluginUtilsTest, CreateFactoriesRejectsInconsistentFactoryOutputs) {
  std::vector<ep_library_plugin_utils::OrtEpFactoryUniquePtr> factories;

  const auto status = ep_library_plugin_utils::CreateFactories(
      ReturnInconsistentFactoryOutputs, ReleaseTestFactory, "test plugin", factories);

  EXPECT_FALSE(status.IsOK());
  EXPECT_TRUE(factories.empty());
  EXPECT_EQ(released_factory_count, 1u);
}

}  // namespace onnxruntime::test
