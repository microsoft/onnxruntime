// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/abi_devices.h"
#include "test/autoep/test_autoep_utils.h"

using namespace onnxruntime;

TEST(EpCompatibilitySelectBestTest, SelectBestCompiledModelCompatibilityInfo_UsesHardwareDevices) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpCompatSelectBestExampleEp"};

  onnxruntime::test::RegisteredEpDeviceUniquePtr example_ep;
  onnxruntime::test::Utils::RegisterAndGetExampleEp(
      env, onnxruntime::test::Utils::example_ep_info, example_ep);

  ASSERT_NE(example_ep.get(), nullptr);

  const OrtEpDevice* ep_device = example_ep.get();
  OrtEpFactory* factory = ep_device->GetMutableFactory();
  ASSERT_NE(factory, nullptr);
  ASSERT_NE(factory->SelectBestCompiledModelCompatibilityInfo, nullptr);

  // API takes OrtHardwareDevice[].
  const OrtHardwareDevice* devices[] = {ep_device->device};

  // Match ExampleEpFactory::ValidateCompiledModelCompatibilityInfoImpl format:
  // "<ep_name>;version=X;ort_api_version=Y;hardware_architecture=Z"
  const std::string ep_name = factory->GetName(factory);

  const std::string optimal =
      ep_name + ";version=0.1.0;ort_api_version=" + std::to_string(ORT_API_VERSION) + ";hardware_architecture=arch1";

  const std::string prefer_recompile =
      ep_name + ";version=9.9.9;ort_api_version=" + std::to_string(ORT_API_VERSION) + ";hardware_architecture=arch1";

  const std::string unsupported =
      "SomeOtherEp;version=0.1.0;ort_api_version=" + std::to_string(ORT_API_VERSION) + ";hardware_architecture=arch1";

  const char* compatibility_infos[] = {
      unsupported.c_str(),
      prefer_recompile.c_str(),
      optimal.c_str(),
  };

  size_t selected_index = std::numeric_limits<size_t>::max();
  OrtStatus* st = factory->SelectBestCompiledModelCompatibilityInfo(
      factory,
      devices,
      1,
      compatibility_infos,
      3,
      &selected_index);

  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);
  ASSERT_EQ(st, nullptr) << (st ? api->GetErrorMessage(st) : "");

  // Best candidate should be the fully optimal one.
  EXPECT_EQ(selected_index, 2u);
}