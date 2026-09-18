// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/inlined_containers.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/ort_value.h"
#include "core/framework/plugin_data_transfer.h"
#include "core/framework/stream_handles.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

TEST(DataTransferManagerTest, PluginCopiesForwardStreams) {
  struct TestDataTransfer final : OrtDataTransferImpl {
    TestDataTransfer() : OrtDataTransferImpl{} {
      ort_version_supported = ORT_API_VERSION;
      Release = [](OrtDataTransferImpl*) noexcept {};
      CopyTensors = [](OrtDataTransferImpl* impl, const OrtValue**, OrtValue**,
                       OrtSyncStream** streams, size_t num_tensors) noexcept -> OrtStatus* {
        auto& self = *static_cast<TestDataTransfer*>(impl);
        self.copied_tensors += num_tensors;
        self.last_stream = streams != nullptr && num_tensors > 0 ? streams[0] : nullptr;
        return nullptr;
      };
    }

    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestDataTransfer);

    size_t copied_tensors = 0;
    OrtSyncStream* last_stream = nullptr;
  } impl;

  plugin_ep::DataTransfer data_transfer{impl};
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  Tensor source{DataTypeImpl::GetType<float>(), TensorShape{4}, allocator};
  Tensor destination{DataTypeImpl::GetType<float>(), TensorShape{4}, allocator};
  OrtDevice device;
  Stream stream{nullptr, device};

  ASSERT_STATUS_OK(data_transfer.CopyTensorAsync(source, destination, stream));
  EXPECT_EQ(impl.copied_tensors, 1U);
  EXPECT_EQ(impl.last_stream, reinterpret_cast<OrtSyncStream*>(&stream));

  ASSERT_STATUS_OK(data_transfer.CopyTensor(source, destination));
  EXPECT_EQ(impl.copied_tensors, 2U);
  EXPECT_EQ(impl.last_stream, nullptr);

  ASSERT_STATUS_OK(data_transfer.CopyTensors({{source, destination, &stream}}));
  EXPECT_EQ(impl.copied_tensors, 3U);
  EXPECT_EQ(impl.last_stream, reinterpret_cast<OrtSyncStream*>(&stream));
}

// DataTransferManager::CopyTensors should validate sizes match before calling the IDataTransfer implementation
TEST(DataTransferManagerTest, BatchedTensorCopyBadSize) {
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  std::vector<OrtValue> src_tensors{2};
  InlinedVector<int64_t> shape_a{4}, shape_b{5}, shape_c{6};
  std::vector<OrtValue> dst_tensors{2};

  // first pair is matched
  AllocateMLValue<float>(allocator, shape_a, &src_tensors[0]);
  AllocateMLValue<float>(allocator, shape_a, &dst_tensors[0]);

  // second pair has size mismatch
  AllocateMLValue<float>(allocator, shape_c, &src_tensors[1]);
  AllocateMLValue<float>(allocator, shape_b, &dst_tensors[1]);

  DataTransferManager dtm;
  ASSERT_STATUS_OK(dtm.RegisterDataTransfer(std::make_unique<CPUDataTransfer>()));

  std::vector<IDataTransfer::SrcDstPair> src_dst_pairs;
  src_dst_pairs.push_back({src_tensors[0].Get<Tensor>(), *dst_tensors[0].GetMutable<Tensor>(), nullptr});
  src_dst_pairs.push_back({src_tensors[1].Get<Tensor>(), *dst_tensors[1].GetMutable<Tensor>(), nullptr});
  auto status = dtm.CopyTensors(src_dst_pairs);

  ASSERT_STATUS_NOT_OK(status);
  ASSERT_THAT(status.ErrorMessage(), testing::HasSubstr("Tensor size mismatch"));
}

}  // namespace test
}  // namespace onnxruntime
