// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <string>

#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace_bounds.h"
#include "core/framework/allocator.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

TEST(GroupQueryAttentionHelperTest, PromptClassificationAndAdmissionRespectTotalLengthResidency) {
  const auto allocator = CPUAllocator::DefaultInstance();
  struct Case {
    int batch;
    int sequence;
    int total;
    bool has_past;
  };
  for (const auto& test_case : {Case{1, 1, 1, false}, Case{1, 4, 4, false},
                                Case{1, 1, 1, true}, Case{1, 4, 4, true},
                                Case{1, 1, 8, true}, Case{1, 4, 8, true},
                                Case{2, 4, 4, true}, Case{2, 4, 8, true}}) {
    for (bool device_total : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "batch=" << test_case.batch << " sequence=" << test_case.sequence
                   << " total=" << test_case.total << " past=" << test_case.has_past
                   << " device_total=" << device_total);
      Tensor query(DataTypeImpl::GetType<float>(),
                   TensorShape{test_case.batch, test_case.sequence, 8}, allocator);
      Tensor key(DataTypeImpl::GetType<float>(), query.Shape(), allocator);
      Tensor value(DataTypeImpl::GetType<float>(), query.Shape(), allocator);
      Tensor past_key(DataTypeImpl::GetType<float>(), TensorShape{test_case.batch, 1, 16, 8}, allocator);
      Tensor past_value(DataTypeImpl::GetType<float>(), past_key.Shape(), allocator);
      Tensor lengths(DataTypeImpl::GetType<int32_t>(), TensorShape{test_case.batch}, allocator);
      int32_t total = test_case.total;
      const OrtMemoryInfo device_info(
          "Cuda", OrtDeviceAllocator,
          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0),
          OrtMemTypeDefault);
      // Only the scalar's location metadata is needed; validation must not read device data.
      Tensor total_length(DataTypeImpl::GetType<int32_t>(), TensorShape{},
                          &total, device_total ? device_info : allocator->Info());
      contrib::GroupQueryAttentionParameters parameters{};
      const auto status = contrib::group_query_attention_helper::CheckInputs<Tensor>(
          &query, &key, &value,
          test_case.has_past ? &past_key : nullptr, test_case.has_past ? &past_value : nullptr,
          nullptr, nullptr, &parameters, 1, 1, &lengths, &total_length, 0.0f, 0.0f, 0);
      const bool cpu_chunk = test_case.sequence > 1 && test_case.sequence != test_case.total;
      if (!device_total && cpu_chunk && test_case.batch != 1) {
        ASSERT_FALSE(status.IsOK());
        EXPECT_NE(status.ErrorMessage().find("batch_size must be 1"), std::string::npos);
        continue;
      }
      ASSERT_STATUS_OK(status);
      EXPECT_EQ(parameters.is_first_prompt, !device_total && test_case.sequence == test_case.total);
      EXPECT_EQ(parameters.is_subsequent_prompt, !device_total && cpu_chunk);
      const auto phase = contrib::cuda::ClassifyGQAExecutionPhase(
          parameters.is_first_prompt, parameters.sequence_length);
      for (int64_t sequence_bound : {int64_t{test_case.sequence}, int64_t{8}}) {
        EXPECT_TRUE(contrib::cuda::IsGQAPhaseReachableForSequenceBound(phase, sequence_bound))
            << "sequence bound=" << sequence_bound;
      }
      if (device_total) {
        EXPECT_EQ(phase, test_case.sequence == 1 ? contrib::cuda::GQAExecutionPhase::Decode
                                                 : contrib::cuda::GQAExecutionPhase::ChunkedPrefill);
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
