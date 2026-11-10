// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/session/onnxruntime_c_api.h"

TEST(StatefulAvailabilityTest, PublicApiMatchesBuildConfiguration) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);
#if defined(ORT_ENABLE_STATEFUL)
  EXPECT_NE(api->GetStatefulApi(), nullptr);
#else
  EXPECT_EQ(api->GetStatefulApi(), nullptr);
  const OrtModelPackageApi* package_api = api->GetModelPackageApi();
  ASSERT_NE(package_api, nullptr);
  int sentinel = 0;
  auto* expected = reinterpret_cast<OrtSessionChain*>(&sentinel);
  OrtSessionChain* result = expected;
  OrtModelPackageChainConfig config{};
  OrtStatus* status = package_api->CreateSessionChain(nullptr, nullptr, &config, &result);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(api->GetErrorCode(status), ORT_NOT_IMPLEMENTED);
  EXPECT_EQ(result, expected);
  api->ReleaseStatus(status);
#endif
}
