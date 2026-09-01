// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/contrib_ops/matmul_nbits_prepack_sharing_test_util.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/tensor.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

void CheckSharedPrepackedWeights(OpTester& test, PrepackSharingMode mode,
                                 const std::vector<int64_t>& b_dims,
                                 std::vector<uint8_t>& b_data) {
  SessionOptions so;
  OrtValue b_ortvalue;

  switch (mode) {
    case PrepackSharingMode::kAddInitializer:
    case PrepackSharingMode::kAddInitializerExpectNoPrepack:
      // Register B as an explicitly shared initializer (the pre-existing sharing mechanism).
      Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape(b_dims), b_data.data(),
                           OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b_ortvalue);
      ASSERT_STATUS_OK(so.AddInitializer("B", &b_ortvalue));
      break;
    case PrepackSharingMode::kNoSharing:
      // Neither opt-in mechanism is used.
      break;
  }

  // Have all sessions created by this OpTester use the same pre-packed weights container.
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited to the CPU EP, so the sharing behavior is only exercised there.
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &ep_vec, {},
             &number_of_pre_packed_weights_counter_session_1,
             &number_of_shared_pre_packed_weights_counter);
    // Nothing can be shared yet because this is the first session.
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  const auto number_of_elements_in_shared_container = test.GetNumPrePackedWeightsShared();

  if (mode == PrepackSharingMode::kAddInitializerExpectNoPrepack) {
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, static_cast<size_t>(0));
    ASSERT_EQ(number_of_elements_in_shared_container, static_cast<size_t>(0));

    {
      size_t number_of_pre_packed_weights_counter_session_2 = 0;
      auto ep_vec = cpu_ep();
      test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &ep_vec, {},
               &number_of_pre_packed_weights_counter_session_2,
               &number_of_shared_pre_packed_weights_counter);
      ASSERT_EQ(number_of_pre_packed_weights_counter_session_2, static_cast<size_t>(0));
      ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
      ASSERT_EQ(test.GetNumPrePackedWeightsShared(), static_cast<size_t>(0));
    }
    return;
  }

  if (mode == PrepackSharingMode::kNoSharing) {
    // Without opting in, pre-packed weights must not be placed in the shared container.
    ASSERT_EQ(number_of_elements_in_shared_container, static_cast<size_t>(0));
  }

  // On some platforms/architectures MLAS may choose not to pre-pack, in which case there is nothing
  // to share and we cannot meaningfully continue.
  if (number_of_pre_packed_weights_counter_session_1 == 0) {
    return;
  }

  if (mode != PrepackSharingMode::kNoSharing) {
    // At least the quantized weight B is content-addressed into the shared container. Some
    // architectures (e.g. ARM64 KleidiAI) additionally pre-pack scales, but in the AddInitializer
    // mode only the explicitly-registered B participates, so the container can hold fewer elements
    // than the total number of pre-packed weights.
    ASSERT_GT(number_of_elements_in_shared_container, static_cast<size_t>(0));
    ASSERT_LE(number_of_elements_in_shared_container, number_of_pre_packed_weights_counter_session_1);
  }

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &ep_vec, {},
             &number_of_pre_packed_weights_counter_session_2,
             &number_of_shared_pre_packed_weights_counter);

    // The same number of weights is pre-packed in both sessions.
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Every weight stored in the shared container is served from it (i.e. shared) in the second
    // session. For the no-sharing control this is zero; otherwise it matches the container size.
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, number_of_elements_in_shared_container);

    if (mode == PrepackSharingMode::kNoSharing) {
      ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
    } else {
      ASSERT_GT(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
