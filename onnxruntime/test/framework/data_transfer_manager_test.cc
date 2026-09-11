// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/inlined_containers.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/ort_value.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

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
