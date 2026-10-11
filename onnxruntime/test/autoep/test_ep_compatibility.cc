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

TEST(EpCompatibilitySelectBestTest, SelectBestModelCandidate_UsesHardwareDevices) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpCompatSelectBestExampleEp"};

  onnxruntime::test::RegisteredEpDeviceUniquePtr example_ep;
  onnxruntime::test::Utils::RegisterAndGetExampleEp(
      env, onnxruntime::test::Utils::example_ep_info, example_ep);

  ASSERT_NE(example_ep.get(), nullptr);

  const OrtEpDevice* ep_device = example_ep.get();
  OrtEpFactory* factory = ep_device->GetMutableFactory();
  ASSERT_NE(factory, nullptr);
  ASSERT_NE(factory->SelectBestModelCandidate, nullptr);

  const std::string ep_name = factory->GetName(factory);

  const std::string optimal =
      ep_name + ";version=0.1.0;ort_api_version=" + std::to_string(ORT_API_VERSION) + ";hardware_architecture=arch1";

  const std::string prefer_recompile =
      ep_name + ";version=9.9.9;ort_api_version=" + std::to_string(ORT_API_VERSION) + ";hardware_architecture=arch1";

  const std::string unsupported =
      "SomeOtherEp;version=0.1.0;ort_api_version=" + std::to_string(ORT_API_VERSION) + ";hardware_architecture=arch1";

  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtKeyValuePairs* kvp0 = nullptr;
  OrtKeyValuePairs* kvp1 = nullptr;
  OrtKeyValuePairs* kvp2 = nullptr;
  api->CreateKeyValuePairs(&kvp0);
  api->CreateKeyValuePairs(&kvp1);
  api->CreateKeyValuePairs(&kvp2);
  api->AddKeyValuePair(kvp0, "ep_compatibility_info", unsupported.c_str());
  api->AddKeyValuePair(kvp1, "ep_compatibility_info", prefer_recompile.c_str());
  api->AddKeyValuePair(kvp2, "ep_compatibility_info", optimal.c_str());

  const OrtKeyValuePairs* candidates[] = {kvp0, kvp1, kvp2};

  size_t selected_index = std::numeric_limits<size_t>::max();
  OrtStatus* st = factory->SelectBestModelCandidate(
      factory, ep_device->device, candidates, 3, nullptr, &selected_index);

  ASSERT_EQ(st, nullptr) << (st ? api->GetErrorMessage(st) : "");

  EXPECT_EQ(selected_index, 2u);

  api->ReleaseKeyValuePairs(kvp0);
  api->ReleaseKeyValuePairs(kvp1);
  api->ReleaseKeyValuePairs(kvp2);
}